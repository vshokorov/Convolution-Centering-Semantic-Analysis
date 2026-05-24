import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed
from tqdm import trange

sys.path.append('./')

from backbones import get_model, replace_bn_with
from my_help_functions.hooks import register_hooks, fuse_bn_running_var
from my_help_functions.utils import get_sampled_cos_paired, ScalableMask
from utils.utils_callbacks import CallBackLogging, CallBackVerification


rank = 0
local_rank = 0
world_size = 1
distributed.init_process_group(
    backend="nccl",
    init_method="tcp://127.0.0.1:1241",
    rank=rank,
    world_size=world_size,
)


def get_and_save_res(full_data, issame, ds_name, layer_data_name, layer_f_builder):
    print('run get_and_save_res for', layer_data_name, ds_name)
    res = defaultdict(lambda: defaultdict(lambda: np.full((NUM_CLASSES, NUM_CLASSES), np.nan)))
    full_data = full_data.cuda()
    assert full_data.size(0) == issame.size(0) * 2

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
        TE = 2  # if OOM, set more than 1
        for batch_id in range(0, full_data.size(0), batch_size):
            assert batch_id % 2 == 0
            imgs = full_data[batch_id:batch_id + batch_size]
            with torch.no_grad():
                output = model(imgs)

            if layer_input_fwd[name1] is None:
                continue

            if all_features_0 is None:
                if (
                    layer_input_fwd[name1][0].element_size()
                    * layer_input_fwd[name1][0][0].numel()
                    * full_data.size(0)
                    > DOWNSAMPLE_CONFIG['max_gb'] * 1024 ** 3
                ):
                    TE = DOWNSAMPLE_CONFIG['take_every']
                    print(f'Set take_every={TE}: {name1}, due to memory limit...')

                all_features_0 = torch.empty(
                    (full_data.size(0), *layer_input_fwd[name1][0][..., ::TE, ::TE].shape[1:]),
                    dtype=layer_input_fwd[name1][0].dtype,
                    device=F_DEVICE,
                )
                if layer_input_fwd[name1][1] is not None:
                    all_features_1 = torch.empty(
                        (full_data.size(0), *layer_input_fwd[name1][1][..., ::TE, ::TE].shape[1:]),
                        dtype=layer_input_fwd[name1][1].dtype,
                        device=F_DEVICE,
                    )

            all_features_0[batch_id:batch_id + batch_size] = layer_input_fwd[name1][0][..., ::TE, ::TE].to(F_DEVICE)
            if layer_input_fwd[name1][1] is not None:
                all_features_1[batch_id:batch_id + batch_size] = layer_input_fwd[name1][1][..., ::TE, ::TE].to(F_DEVICE)

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
        mask_indices_l1 = torch.nonzero(mask[:2])
        mask_indices_l2 = torch.nonzero(mask[2:])

        all_r = get_sampled_cos_paired(
            issame,
            mask_indices_l1,
            mask_indices_l2,
            all_features_0,
            all_features_1,
            'var_mean',
            fix_order=True,
        )
        l1 = 0
        try:
            for k_r, r in all_r.items():
                if k_r == 'pos':
                    l2 = 0
                elif k_r == 'neg':
                    l2 = 1
                else:
                    raise ValueError(k_r)
        except AttributeError:
            del all_features_0
            del all_features_1
            torch.cuda.empty_cache()
            continue

        if r is not None:
            if any(torch.isnan(v) for v in r.values()):
                if not (ds_name.endswith('_back') and k_r == 'pos'):
                    print(name1, k_r, r)
            for k, v in r.items():
                res[name1][k][l1, l2] = v.item() + np.nan_to_num(res[name1][k][l1, l2])
            res[name1]['count'][l1, l2] = 1 + np.nan_to_num(res[name1]['count'][l1, l2])

        del all_features_0
        del all_features_1
        torch.cuda.empty_cache()
    del full_data

    for n in res.keys():
        res[n] = {k: v / res[n]['count'] for k, v in res[n].items()}

    path = f"heap/arcface/{ds_name}/{MODEL_ARCH}/"
    os.makedirs(path, exist_ok=True)
    full_file_name = os.path.join(path, layer_data_name.replace(' ', '_') + '.pkl')
    print('save at', full_file_name)
    with open(full_file_name, 'wb') as fw:
        pickle.dump(dict(res), fw)


def merge_background_angles(ds_name):
    PATH = f'heap/arcface/{ds_name}/{MODEL_ARCH}'
    PATH_B = f'heap/arcface/{ds_name}_back/{MODEL_ARCH}'
    for f in os.listdir(PATH):
        with open(os.path.join(PATH, f), 'rb') as fr:
            res = pickle.load(fr)

        if not os.path.exists(os.path.join(PATH_B, f)):
            print('Skip', os.path.join(PATH_B, f), '-- no file')
            continue
        with open(os.path.join(PATH_B, f), 'rb') as fr:
            res_b = pickle.load(fr)

        for ln in res:
            for k in res[ln]:
                old_m = res[ln][k]
                assert old_m.shape == (2, 2)
                m = np.full((old_m.shape[0] + 1, old_m.shape[1] + 1), np.nan)
                m[:old_m.shape[0], :old_m.shape[1]] = old_m
                m[:old_m.shape[0], old_m.shape[1]] = res_b[ln][k][:, 1]
                res[ln][k] = m

        with open(os.path.join(PATH, f), 'wb') as fw:
            pickle.dump(res, fw)
        os.remove(os.path.join(PATH_B, f))
    os.rmdir(PATH_B)


def _merge_all_datasets():
    for ds_k in DATASETS:
        if not ds_k.endswith('_back'):
            merge_background_angles(ds_k)


def run_experiment(layer_data_name, layer_f_builder):
    for ds_k, ds_v in DATASETS.items():
        get_and_save_res(*ds_v, ds_name=ds_k, layer_data_name=layer_data_name, layer_f_builder=layer_f_builder)
    _merge_all_datasets()


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
            shift += (l.bias.detach() / l.weight.detach()).to(output.dtype)
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
            shift += (l.bias.detach() / l.weight.detach()).to(output.dtype)
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
                for ds_k, ds_v in DATASETS.items():
                    get_and_save_res(
                        *ds_v,
                        ds_name=ds_k,
                        layer_data_name=f'While Conv with S {corr_type}',
                        layer_f_builder=layer_f_fig8(corr_type),
                    )
    _merge_all_datasets()


EXPERIMENT_REGISTRY = {
    'After conv': lambda: run_experiment('After conv', layer_f_after_conv),
    'Before conv vs After conv': lambda: run_experiment('Before conv vs After conv', layer_f_before_conv_vs_after_conv),
    'After shifting': lambda: run_experiment('After shifting', layer_f_after_shifting),
    'Before shifting vs After shifting': lambda: run_experiment('Before shifting vs After shifting', layer_f_before_shifting_vs_after_shifting),
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
    NUM_CLASSES = 2
    MODEL_ARCH = 'r50'
    DATA_LIMIT = 10000
    DOWNSAMPLE_CONFIG = {
        'max_gb': 1000,
        'take_every': 2,
    }

    args = parse_args()

    model = get_model(MODEL_ARCH, dropout=0.0, fp16=True, num_features=512).cuda(0)
    model_weights = torch.load(f'./facerec_weights/PartialFC_{args.replace_bn_with}/model.pt')
    replace_bn_with(model, args.replace_bn_with)
    model = model.cuda()
    MODEL_ARCH = MODEL_ARCH + '_' + args.replace_bn_with
    print('run', MODEL_ARCH)
    model.load_state_dict(model_weights)
    model = fuse_bn_running_var(model.eval())

    val_targets = ['lfw']
    rec = "./datasets/"

    class SW:
        @staticmethod
        def add_scalar(**kwargs):
            print('SW:', kwargs)

    callback_verification = CallBackVerification(
        val_targets=val_targets, rec_prefix=rec,
        summary_writer=SW(), wandb_logger=None,
    )

    DATASETS = {
        'lfw': (
            torch.from_numpy(callback_verification.ver_list[0][0][:])[:DATA_LIMIT, [2, 1, 0]],
            torch.from_numpy(callback_verification.ver_list[0][1][:DATA_LIMIT // 2]),
        )
    }

    top_svd_data = DATASETS['lfw'][0].reshape(100, 100, 3, 112, 112).mean(1)
    assert DATA_LIMIT % (2 * top_svd_data.size(0)) == 0
    DATASETS.update({
        k + '_back': (
            torch.cat([
                d[:DATA_LIMIT:2],
                top_svd_data.repeat((DATA_LIMIT // (2 * top_svd_data.size(0)), 1, 1, 1)),
            ], dim=1).view(-1, 3, 112, 112),
            torch.zeros(DATA_LIMIT // 2, dtype=torch.bool),
        )
        for k, (d, _) in DATASETS.items()
    })

    scalable_mask = ScalableMask((4, 112, 112), padding_p=0.2, device='cuda')
    scalable_mask.orig_mask[1:3] = 0

    names = load_requested_names(args)
    
    path = f'heap/arcface/lfw/{MODEL_ARCH}/'
    
    print("experiments: ", names)
    
    for name in names:
        if name == 'Fig8':
            all_exist = True
            fig8_files = []
            for v in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
                for t in ['top', 'low']:
                    for s in ['center', 's']:
                        corr_type = f'{t}_by_{s}_{v}_norm'
                        filename = path + f'While Conv with S {corr_type}'.replace(' ', '_') + '.pkl'
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
