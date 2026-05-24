import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import pickle
import re
import sys
from collections import defaultdict
from easydict import EasyDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed
from tqdm import trange

sys.path.append('./')

from my_help_functions.hooks import register_hooks, fuse_bn_running_var
from my_help_functions.utils import get_sampled_cos, ScalableMask
from pytorch_cifar100.train import cifar100_test_dataset as test_dataset
from pytorch_cifar100.train import FacenDataset
from pytorch_cifar100.utils import get_network


def get_and_save_res(layer_data_name, layer_f_builder):
    print('run get_and_save_res for ', layer_data_name)
    res = defaultdict(lambda : defaultdict(lambda : np.full((NUM_CLASSES, NUM_CLASSES), np.nan)))
    assert full_data.size(0) == full_labels.size(0)


    modules = list(model.named_modules())
    for i in trange(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]

        if not (isinstance(layer1, torch.nn.Conv2d) and isinstance(layer2, torch.nn.BatchNorm2d)):
            continue
        assert layer1.bias is None

        hooks, layer_input_fwd = register_hooks(model, [name1], layer_f_builder(layer2))
        all_features_0 = None
        all_features_1 = None

        batch_size = 512
        for batch_id in range(0, full_data.size(0), batch_size):
            imgs = full_data[batch_id:batch_id+batch_size].cuda()
            with torch.no_grad():
                output = model(imgs)
            
            if layer_input_fwd[name1] is None:
                continue
        
            if all_features_0 is None:
                all_features_0 = torch.empty(
                    (full_data.size(0), *layer_input_fwd[name1][0].shape[1:]),
                    dtype=layer_input_fwd[name1][0].dtype,
                    device=F_DEVICE
                )
                if layer_input_fwd[name1][1] is not None:
                    all_features_1 = torch.empty(
                        (full_data.size(0), *layer_input_fwd[name1][1].shape[1:]),
                        dtype=layer_input_fwd[name1][1].dtype,
                        device=F_DEVICE
                    )

            all_features_0[batch_id:batch_id+batch_size] = layer_input_fwd[name1][0].to(F_DEVICE)
            if layer_input_fwd[name1][1] is not None:
                all_features_1[batch_id:batch_id+batch_size] = layer_input_fwd[name1][1].to(F_DEVICE)

        for h in hooks.values(): 
            h.remove()
        hooks.clear()
        del layer_input_fwd
        torch.cuda.empty_cache()

        if all_features_0 is None:
            continue

        all_features_0 = all_features_0.cuda()
        if all_features_1 is not None:
            all_features_1 = all_features_1.cuda()

        mask = scalable_mask.create_mask((all_features_0.size(2), all_features_0.size(3)))
        batch_unique_labels = torch.unique(full_labels).numpy()
        for i, l1 in enumerate(batch_unique_labels):
            mask_l1 = torch.zeros(all_features_0.size(0), all_features_0.size(2), all_features_0.size(3), dtype=bool, device='cuda')
            mask_l1[full_labels == l1] = 1
            mask_l1[~mask.expand(all_features_0.size(0), -1, -1)] = 0
            mask_indices_l1 = torch.nonzero(mask_l1)
            
            for l2 in batch_unique_labels[i:]:
                mask_l2 = torch.zeros_like(mask_l1)
                mask_l2[full_labels == l2] = 1
                mask_l2[~mask.expand(all_features_0.size(0), -1, -1)] = 0
                mask_indices_l2 = torch.nonzero(mask_l2)
                if l1 == l2:
                    l2_n_images = (full_labels == l2).sum()
                    mask_indices_l2 = mask_indices_l2.view(l2_n_images, mask.sum(), 3)[torch.randperm(l2_n_images)].reshape(-1, 3)
                
                r = get_sampled_cos(
                    mask_indices_l1, 
                    mask_indices_l2, 
                    all_features_0,
                    all_features_1,
                    'var_mean',
                    fix_order=True,
                )
                r = {k: v.item() for k, v in r.items()}
                if not r is None:
                    if any(np.isnan(v) for v in r.values()):
                        print(r, name1)
                    for k, v in r.items():
                        res[name1][k][l1, l2] = v + np.nan_to_num(res[name1][k][l1, l2])
                    res[name1]['count'][l1, l2] = 1 + np.nan_to_num(res[name1]['count'][l1, l2])
        
        del all_features_0
        del all_features_1
        torch.cuda.empty_cache()

    for n in res.keys():
        res[n] = {k: v / res[n]['count'] for k, v in res[n].items()}
    
    path = f"heap/cifar100/{MODEL_ARCH}/"
    os.makedirs(path, exist_ok=True)
    full_file_name = os.path.join(path, layer_data_name.replace(' ', '_') + '.pkl')
    print('save at', full_file_name)
    with open(full_file_name, 'wb') as fw:
        pickle.dump(dict(res), fw)


def layer_f_after_conv(l):
    def foo(module, input, output):
        return output, None
    return foo


def layer_f_after_shifting(l):
    def foo(module, input, output):
        shift = 0
        if l.running_mean is not None:
            shift -= l.running_mean.to(output.dtype)
        if l.bias is not None:
            shift += (l.bias.detach()).to(output.dtype)
        if isinstance(shift, int):
            return None
        output_ = output + shift[None, :, None, None]
        return output_, None
    return foo


def layer_f_before_conv_vs_after_conv(l):
    def foo(module, input, output):
        if input.size(2) != output.size(2) or input.size(3) != output.size(3):
            return None
        return input, output
    return foo


def layer_f_before_shifting_vs_after_shifting(l):
    def foo(module, input, output):
        if input.size(2) != output.size(2) or input.size(3) != output.size(3):
            return None
        shift = 0
        if l.running_mean is not None:
            shift -= l.running_mean.to(output.dtype)
        if l.bias is not None:
            shift += (l.bias.detach()).to(output.dtype)
        if isinstance(shift, int):
            return None
        output_ = output + shift[None, :, None, None]
        return output, output_
    return foo


def layer_f_fig8(corr_type):
    def builder(next_bn):
        def foo(module, input, output):
            if input.size(2) != output.size(2) or input.size(3) != output.size(3):
                return None
            if module.weight.numel() / module.weight.size(0) < module.weight.size(0):
                return None

            w_pre = module.weight.detach().permute(0, 2, 3, 1)
            try:
                svd_out = torch.linalg.svd(w_pre.flatten(1), full_matrices=False)
                U, S, Vh = svd_out.U, svd_out.S, svd_out.Vh
            except AttributeError:
                U, S, V = torch.svd(w_pre.flatten(1))
                Vh = V.T

            Vh_weight = Vh.reshape(w_pre.shape).permute(0, 3, 1, 2)
            center_after_s = U.T @ next_bn.running_mean

            mask_to_zero = torch.ones_like(center_after_s, dtype=torch.bool)

            m = re.match(r'(top|low)_by_(center|s)_(0\.\d+)_(norm|count)', corr_type)
            if m is None:
                raise ValueError(corr_type)

            t = center_after_s.abs() if m.group(2) == 'center' else S
            idx = t.sort(descending=(m.group(1) == 'top')).indices
            v = S[idx]

            if m.group(4) == 'norm':
                k = torch.where(
                    torch.cumsum(v.pow(2), dim=0).sqrt() > v.norm() * float(m.group(3))
                )[0][0] + 1
                mask_to_zero[idx[:k]] = 0
            else:
                mask_to_zero[idx[:int(idx.size(0) * float(m.group(3)))]] = 0

            assert not mask_to_zero.all()
            Vh_weight[mask_to_zero] = 0
            Vh_weight *= S[:, None, None, None]

            x = F.conv2d(
                input.detach(),
                Vh_weight,
                bias=None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
            )
            return input.detach(), x

        return foo
    return builder

def run_fig8():
    assert args.replace_bn_with == 'mean_bias', (
        f"Fig8 requires replace_bn_with='mean_bias', got '{args.replace_bn_with}'"
    )
    for v in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
        for t in ['top', 'low']:
            for s in ['center', 's']:
                corr_type = f'{t}_by_{s}_{v}_norm'
                get_and_save_res(
                f'Fig8 While Conv {corr_type}',
                layer_f_fig8(corr_type)
                )


EXPERIMENT_REGISTRY = {
    'After conv': lambda: get_and_save_res('After conv', layer_f_after_conv),
    'Before conv vs After conv': lambda: get_and_save_res('Before conv vs After conv', layer_f_before_conv_vs_after_conv),
    'After shifting': lambda: get_and_save_res('After shifting', layer_f_after_shifting),
    'Before shifting vs After shifting': lambda: get_and_save_res('Before shifting vs After shifting', layer_f_before_shifting_vs_after_shifting),
    'Fig8': run_fig8,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute mutual distance statistics across ArcFace model layers.'
    )

    parser.add_argument(
        'replace_bn_with',
        choices=['mean_bias', 'only_mean', 'only_bias', 'nothing'],
        help=(
            'How BatchNorm was replaced during training'
            'Note: Fig8 experiment requires "mean_bias".'
        ),
    )

    parser.add_argument(
        '--data_names',
        type=str,
        help=(
            'Comma-separated list of experiment names to run. '
            f'Available: {", ".join(EXPERIMENT_REGISTRY)}.'
        ),
    )

    parser.add_argument(
        '--data_names_file',
        type=str,
        help='Path to a .txt file with one experiment name per line.',
    )

    return parser.parse_args()


def load_requested_names(args):
    names = []

    if args.data_names:
        names.extend(n.strip() for n in args.data_names.split(','))

    if args.data_names_file:
        with open("tools/figures_files/" + args.data_names_file) as f:
            names.extend(line.strip() for line in f if line.strip())

    if not names:
        raise ValueError(
            'No experiment names provided. Use --data_names or --data_names_file. '
            f'Available: {list(EXPERIMENT_REGISTRY)}.'
        )

    unknown = [x for x in names if x not in EXPERIMENT_REGISTRY]
    if unknown:
        raise ValueError(f'Unknown experiment names: {unknown}. Available: {list(EXPERIMENT_REGISTRY)}.')

    return list(dict.fromkeys(names))


if __name__ == '__main__':
    F_DEVICE = torch.device('cuda')

    args = parse_args()

    MODEL_ARCH = f'resnet50_{args.replace_bn_with}'
    CHECKPOINT_PATH = f'pytorch_cifar100/checkpoint/resnet50_{args.replace_bn_with}-200-regular.pth'
    print('run', MODEL_ARCH)

    NUM_CLASSES = 100
    IMAGE_MASK, PADDING_P_MASK = (1, 32, 32), 0.1
    network_args = EasyDict({'net': MODEL_ARCH, 'gpu': True})
    model = get_network(network_args)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH)
    )
    model = fuse_bn_running_var(model.eval())

    full_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    full_labels = torch.LongTensor([test_dataset[i][1] for i in range(len(test_dataset))])

    full_data = torch.cat([
        full_data, 
        torch.stack([full_data[i:i+100].mean(0) for i in range(0, 10000, 100)])
    ])
    full_labels = torch.cat([full_labels, torch.full((100,), NUM_CLASSES, dtype=int)])
    NUM_CLASSES += 1

    scalable_mask = ScalableMask(IMAGE_MASK, padding_p=PADDING_P_MASK, device='cuda')
    names = load_requested_names(args)
    
    path = f'heap/cifar100/{MODEL_ARCH}/'
    
    print("experiments: ", names)
    
    for name in names:
        if name == 'Fig8':
            all_exist = True
            fig8_files = []
            for v in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
                for t in ['top', 'low']:
                    for s in ['center', 's']:
                        corr_type = f'{t}_by_{s}_{v}_norm'
                        filename = path + f'Fig8 While Conv {corr_type}'.replace(' ', '_') + '.pkl'
                        fig8_files.append(filename)
                        if os.path.exists(filename):
                            print(f'Warning: {filename} already exists!')
                            all_exist = False
            
            if not all_exist:
                response = input('Some Fig8 files already exist. Overwrite all? [y/N]: ')
                if response.lower() != 'y':
                    print('Skipping Fig8...')
                    continue
        else:
            filename = path + name.replace(' ', '_') + '.pkl'
            if os.path.exists(filename):
                print(f'Warning: {filename} already exists!')
                response = input('Overwrite? [y/N]: ')
                if response.lower() != 'y':
                    print(f'Skipping {name}...')
                    continue
        
        EXPERIMENT_REGISTRY[name]()