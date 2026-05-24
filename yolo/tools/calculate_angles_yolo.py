import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import re
import pickle
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import distributed

import sys
sys.path.append('./')

from damo.config.base import parse_config
from damo.dataset import build_dataloader
from damo.utils import fuse_bn_running_var
from tools.demo import Infer

from my_help_functions.datasets import CustomCocoDataset
from my_help_functions.hooks import register_hooks
from my_help_functions.visualise_arch import ArchVisualiser

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12503",
        rank=rank,
        world_size=world_size,
    )


class batch_t: 
    def __init__(self, batch, init_mask=True):
        self.imgs = batch[0].to(device)
        self.targets = batch[1]
        self.img_ids = batch[2]
        self.unique_labels = set()

        MASK_SIZE = (320, 320)
        self.orig_mask = np.zeros(
            (len(config.dataset.class_names), len(self.img_ids), *MASK_SIZE), 
            dtype=np.uint8
        )
        self.orig_back_mask = torch.zeros(
            (len(self.img_ids), *MASK_SIZE), 
            dtype=torch.bool, device='cpu'
        )
        self.orig_back_mask[:, MASK_SIZE[0]//10:-MASK_SIZE[0]//10, MASK_SIZE[1]//10:-MASK_SIZE[1]//10] = 1

        for i, t, in enumerate(self.targets):
            t = t.resize(MASK_SIZE)
            
            for bb, l, seg_mask in zip(t.bbox.long(), t.get_field('labels'), t.get_field('seg_masks').contours):
                if init_mask:
                    mask = np.zeros(t.size, dtype=np.uint8)
                    cv2.fillPoly(mask, pts = [seg_mask.astype(np.int32)], color=(1,))
                    self.orig_mask[l.item()][i] = mask

                self.orig_back_mask[i, bb[1]:bb[3], bb[0]:bb[2]] = 0
                self.unique_labels.add(l.item())

        self.orig_mask = torch.from_numpy(self.orig_mask).to(device) > 0
        self.orig_back_mask = self.orig_back_mask.to(device)

        self.old_size = MASK_SIZE
        self.mask = self.orig_mask
        self.back_mask = self.orig_back_mask
        
        torch.cuda.empty_cache()
    
    def init_mask_by_pos_inds(self, pos_inds, pos_labels):
        self.orig_mask.fill_(0)

        for j in range(len(self)):
            for i in range(3):
                if i == 0:
                    inds_mask = torch.logical_and(j * 8400 <= pos_inds, pos_inds < j * 8400 + 6400)
                    shift, scale = j * 8400, 80
                elif i == 1:
                    inds_mask = torch.logical_and(j * 8400 + 6400 <= pos_inds, pos_inds < j * 8400 + 8000)
                    shift, scale = j * 8400 + 6400, 40
                else:
                    inds_mask = torch.logical_and(j * 8400 + 8000 <= pos_inds, pos_inds < j * 8400 + 8400)
                    shift, scale = j * 8400 + 8000, 20
                
                a = pos_inds[inds_mask]
                a -= shift
                for l in torch.unique(pos_labels[inds_mask]):
                    mask = torch.zeros(scale, scale, dtype=bool, device=device)
                    mask.flatten()[a[pos_labels[inds_mask] == l].cpu()] = 1

                    self.orig_mask[l, j].add_(
                            F.interpolate(
                            mask[None, None].half(), 
                            size=(self.orig_mask.size(2), self.orig_mask.size(3)), 
                            mode='nearest'
                        )[0, 0] > 0
                    )
    
    def _create_mask(self, size):
        del self.mask
        del self.back_mask

        self.mask = torch.zeros(
            (self.orig_mask.size(0), self.orig_mask.size(1), size[0], size[1]), dtype=torch.bool, device=device
        )
        for i in range(self.orig_mask.size(0)):
            self.mask[i] = F.interpolate(self.orig_mask[i][None].half(), size=size, mode='nearest')[0] > 0
        self.back_mask = F.interpolate(self.orig_back_mask[None].half(), size=size, mode='nearest')[0] > 0
        torch.cuda.empty_cache()

        self.old_size = size
    
    def create_mask(self, size):
        if size[0] != self.old_size[0] or size[1] != self.old_size[1]:
            self._create_mask(size)

        return self.mask, self.back_mask
    
    def __len__(self):
        return len(self.img_ids)

def get_sampled_cos(a_indices, b_indices, features, features2=None, reduce=None, n_samples=10000):
    if n_samples > a_indices.size(0) * b_indices.size(0) / 10:
        n_samples = int(a_indices.size(0) * b_indices.size(0) / 10)
        if n_samples < 10:
            return None
    
    a_selected_indices_idx = torch.randint(0, a_indices.size(0), (n_samples,), device=features.device)
    b_selected_indices_idx = torch.randint(0, b_indices.size(0), (n_samples,), device=features.device)

    a_selected_indices = a_indices[a_selected_indices_idx]
    b_selected_indices = b_indices[b_selected_indices_idx]

    def get_cos_for_features(f):
        a = f[a_selected_indices[:, 0], :, a_selected_indices[:, 1], a_selected_indices[:, 2]]
        b = f[b_selected_indices[:, 0], :, b_selected_indices[:, 1], b_selected_indices[:, 2]]
        return torch.sum(
            F.normalize(a, dim=1) * F.normalize(b, dim=1),
            dim=1
        )
    cos = get_cos_for_features(features)
    if not features2 is None:
        cos = get_cos_for_features(features2) - cos

    if reduce is None:
        return cos
    elif reduce == 'mean':
        return {'mean': cos.mean().item()}
    elif reduce == 'var_mean':
        return {k: v.item() for k, v in zip(['var', 'mean'], torch.var_mean(cos))}
    elif reduce == 'quantiles':
        if len(coco_dl) > 1:
            print('WARNING: quantile method is not suitable for batching!')
        qs = torch.Tensor([0.05, 0.25, 0.5, 0.75, 0.95]).to(cos.device)
        return {f'q{q.item():.2f}': v.item() for q, v in zip(qs, torch.quantile(cos, qs))}
    else:
        raise


def get_and_save_res(layer_data_name, layer_f_builder):
    print('run get_and_save_res for', layer_data_name)
    res = defaultdict(lambda : defaultdict(lambda: np.full((81, 81), np.nan)))

    for batch_id, b in enumerate(tqdm(coco_dl)):
        batch = batch_t(b, False)

        model.head.train()
        with torch.no_grad():
            output = model(batch.imgs, [t.to(device) for t in b[1]])
        model.head.eval()
        batch.init_mask_by_pos_inds(output['pos_inds'], output['pos_labels'])

        modules = list(model.named_modules())
        for i in range(len(modules) - 1):
            name1, layer1 = modules[i]
            name2, layer2 = modules[i + 1]

            if not (isinstance(layer1, torch.nn.Conv2d) and isinstance(layer2, torch.nn.BatchNorm2d)):
                continue
            assert layer1.bias is None

            hooks, layer_input_fwd = register_hooks(
                model,
                [name1],
                layer_f_builder(layer2)
            )
            with torch.no_grad():
                output = model(batch.imgs)
            for h in hooks.values(): 
                h.remove()
            hooks.clear()
            
            if layer_input_fwd[name1] is None:
                continue
            mask, back_mask = batch.create_mask((layer_input_fwd[name1][0].size(2), layer_input_fwd[name1][0].size(3)))

            batch_unique_labels = list(batch.unique_labels) + [80]
            for i, l1 in enumerate(batch_unique_labels):
                mask_l1 = back_mask if l1 == 80 else mask[l1]
                mask_indices_l1 = torch.nonzero(mask_l1)
                
                for l2 in batch_unique_labels[i:]:
                    mask_l2 = back_mask if l2 == 80 else mask[l2]
                    mask_indices_l2 = torch.nonzero(mask_l2)
                    
                    r = get_sampled_cos(
                        mask_indices_l1, 
                        mask_indices_l2, 
                        layer_input_fwd[name1][0],
                        layer_input_fwd[name1][1],
                        'var_mean'
                    )
                    if not r is None:
                        if any(np.isnan(v) for v in r.values()):
                            print(r, name1)
                        for k, v in r.items():
                            res[name1][k][l1, l2] = v + np.nan_to_num(res[name1][k][l1, l2])
                        res[name1]['count'][l1, l2] = 1 + np.nan_to_num(res[name1]['count'][l1, l2])
            del layer_input_fwd
            torch.cuda.empty_cache()
        
        del b
        del batch

    for n in res.keys():
        res[n] = {k: v / res[n]['count'] for k, v in res[n].items()}

    path = 'heap/angles/'
    with open(path + layer_data_name.replace(' ', '_') + '.pkl', 'wb') as fw:
        pickle.dump(dict(res), fw)

        
def layer_f_after_conv(l):
    def foo(module, input, output):
        return output, None
    return foo


def layer_f_after_center(l):
    def foo(module, input, output):
        output_ = output - l.running_mean[None, :, None, None]
        return output_, None
    return foo


def layer_f_before_conv_vs_after_conv(l):
    def foo(module, input, output):
        if input.size(2) != output.size(2) or input.size(3) != output.size(3):
            return None
        return input, output
    return foo


def layer_f_before_center_vs_after_center(l):
    def foo(module, input, output):
        if input.size(2) != output.size(2) or input.size(3) != output.size(3):
            return None
        output_ = output - l.running_mean[None, :, None, None]
        return output, output_
    return foo


def layer_f_after_bias(l):
    def foo(module, input, output):
        if input.size(2) != output.size(2) or input.size(3) != output.size(3):
            return None
        output_ = output - l.running_mean[None, :, None, None]
        output_.mul_(l.weight.detach()[None, :, None, None])
        output2 = output_ + l.bias.detach()[None, :, None, None]
        return output2, None
    return foo


def layer_f_before_bias_vs_after_bias(l):
    def foo(module, input, output):
        if input.size(2) != output.size(2) or input.size(3) != output.size(3):
            return None
        output_ = output - l.running_mean[None, :, None, None]
        output_.mul_(l.weight.detach()[None, :, None, None])
        output2 = output_ + l.bias.detach()[None, :, None, None]
        return output_, output2
    return foo


def layer_f_after_shifting(l):
    def foo(module, input, output):
        shift = 0
        if l.running_mean is not None:
            shift -= l.running_mean.to(output.dtype)
        if l.bias is not None:
            shift += l.bias.detach().to(output.dtype)
        if isinstance(shift, int):
            return None
        output_ = output + shift[None, :, None, None]
        return output_, None
    return foo


def layer_f_before_shifting_vs_after_shifting(l):
    def foo(module, input, output):
        if input.size(2) != output.size(2) or input.size(3) != output.size(3):
            return None
        shift = 0
        if l.running_mean is not None:
            shift -= l.running_mean.to(output.dtype)
        if l.bias is not None:
            shift += l.bias.detach().to(output.dtype)
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

            Vh_weight[mask_to_zero] = 0
            Vh_weight *= S[:, None, None, None]

            x = F.conv2d(
                input.detach(),
                Vh_weight,
                bias=None,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation
            )

            return input.detach(), x

        return foo
    return builder


def run_fig8():
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
    'After center': lambda: get_and_save_res('After center', layer_f_after_center),
    'Before conv vs After conv': lambda: get_and_save_res('Before conv vs After conv', layer_f_before_conv_vs_after_conv),
    'Before center vs After center': lambda: get_and_save_res('Before center vs After center', layer_f_before_center_vs_after_center),
    'After bias': lambda: get_and_save_res('After bias', layer_f_after_bias),
    'Before bias vs After bias': lambda: get_and_save_res('Before bias vs After bias', layer_f_before_bias_vs_after_bias),
    'After shifting': lambda: get_and_save_res('After shifting', layer_f_after_shifting),
    'Before shifting vs After shifting': lambda: get_and_save_res('Before shifting vs After shifting', layer_f_before_shifting_vs_after_shifting),
    'Fig8': run_fig8,
}


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_names',
        type=str,
        help='comma-separated list of names'
    )

    parser.add_argument(
        '--data_names_file',
        type=str,
        help='txt file, one name per line'
    )
    
    return parser.parse_args()


def load_requested_names(args):
    names = []

    if args.data_names:
        names.extend(map(lambda n: n.strip(), args.data_names.split(",")))

    if args.data_names_file:
        with open("tools/figures_files/" + args.data_names_file) as f:
            names.extend(line.strip() for line in f if line.strip())

    if not names:
        raise ValueError('empty data names')

    unknown = [x for x in names if x not in EXPERIMENT_REGISTRY]
    if unknown:
        raise ValueError(f'unknown names: {unknown}')

    return list(dict.fromkeys(names))


if __name__ == '__main__':
    config = parse_config('./configs/damoyolo_tinynasL20_T.py')
    device = 'cuda'

    infer_engine = Infer(config, device=device,
        ckpt='./damoyolo_tinynasL20_T_420.pth')

    model = fuse_bn_running_var(infer_engine.model.eval())

    coco_ds = CustomCocoDataset(config, load_train=False)
    coco_dl = build_dataloader(
        [coco_ds],
        config.test.augment,
        len(coco_ds), #//4 + 1, # batch_size
        num_workers=config.miscs.num_workers,
        is_train=False
    )[0]

    args = parse_args()
    names = load_requested_names(args)

    path = 'heap/angles/'
    
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