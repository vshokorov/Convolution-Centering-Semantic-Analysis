from collections import OrderedDict
from typing import Dict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import re


bns = []
convs = []


def remove_all_hooks(model: torch.nn.Module) -> None:
    global bns
    bns = []
    global convs
    convs = []

    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict() # type: ignore
            if hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict() # type: ignore
            if hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict() # type: ignore
            remove_all_hooks(child)


def hook_conv(name, module, input, output):
    global convs
    convs.append([name, module, input[0].squeeze(0), output.squeeze(0), module.weight])


def hook_bn(name, module, input, output):
    global bns
    bns.append([name, module, input[0].squeeze(0), output.squeeze(0)])
    
    
def make_svd_hook(name, module, next_bn, corr_type, with_s):
    def hook(mod, input, output):
        if input[0].size(2) != output.size(2):
            return None
        if module.weight.numel() / module.weight.size(0) < module.weight.size(0):
            return None
        
        inp = input[0].detach()
        w_pre = module.weight.detach().permute(0, 2, 1)
        svd_weight = torch.linalg.svd(w_pre.flatten(1), full_matrices=False)
        Vh_weight = svd_weight.Vh.reshape(w_pre.shape).permute(0, 2, 1)

        center_after_s = (svd_weight.U.T @ next_bn.running_mean)

        mask_to_zero = torch.ones_like(center_after_s, dtype=bool)
        if (m := re.match(r'(top|low)_by_(center|s)_(0\.\d+)_(norm|count)', corr_type)):
            t = center_after_s.abs() if m.groups()[1] == 'center' else svd_weight.S
            i = t.sort(descending = (m.groups()[0] == 'top') ).indices
            v = svd_weight.S[i]

            if m.groups()[3] == 'norm':
                mask_to_zero[i[:torch.where(torch.cumsum(v.pow(2), dim=0).sqrt() > v.norm() * float(m.groups()[2]))[0][0] + 1]] = 0
            else:
                mask_to_zero[i[:int(i.size(0) * float(m.groups()[2]))]] = 0
        else:
            raise ValueError(f'Unknown corr_type: {corr_type}')

        assert not mask_to_zero.all()
        
        Vh_weight[mask_to_zero] = 0
        if with_s:
            Vh_weight *= svd_weight.S[:, None, None]

        x = F.conv1d(inp, Vh_weight, bias=None,
                     stride=module.stride, padding=module.padding, dilation=module.dilation)

        convs.append([inp.cpu(), x.detach().cpu()])
    return hook

    
def register_conv_bn_hooks(model, norm):
    remove_all_hooks(model)
    modules = list(model.named_modules())
    for i in range(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]

        if isinstance(layer1, nn.Conv1d) and isinstance(layer2, norm):
            layer1.register_forward_hook(lambda module, input, output, name=name1: hook_conv(name, module, input, output))
            layer2.register_forward_hook(lambda module, input, output, name=name2: hook_bn(name, module, input, output))
    return bns, convs


def register_hooks_corr_uncorr(model, corr_type, norm, with_s):
    remove_all_hooks(model)
    modules = list(model.named_modules())

    for i in range(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]

        if isinstance(layer1, nn.Conv1d) and isinstance(layer2, norm):
            layer1.register_forward_hook(make_svd_hook(name1, layer1, layer2, corr_type, with_s))

    return convs