import torch
import torch.nn.functional as F
from torch import nn
import warnings


def register_hooks(model, layers_to_add, return_type):
    layer_outputs_fwd = {}
    hooks = {}

    def get_hook(name):
        def hook(module, input, output):
            if callable(return_type):
                layer_outputs_fwd[name] = return_type(module, input[0].detach(), output.detach())#.clone()
            elif return_type == 'input':
                layer_outputs_fwd[name] = input[0].detach()#.clone()
            elif return_type == 'output':
                if (not hasattr(module, 'bias')) or module.bias is None:
                    layer_outputs_fwd[name] = output.detach()#.clone()
                else:
                    output = F.conv2d(input[0].detach(), module.weight, None, module.stride, module.padding)
                    layer_outputs_fwd[name] = output.detach()#.clone()
            elif return_type == 'output_base':
                layer_outputs_fwd[name] = output.detach()#.clone()

            else:
                raise

        return hook
    
    for name, module in model.named_modules():
        if name in layers_to_add:
            hooks[name] = module.register_forward_hook(get_hook(name))
    
    return hooks, layer_outputs_fwd

def fuse_bn_running_var(model, verbose=False):
    
    modules = list(model.named_modules())
    for i in range(len(modules) - 1):
        conv_name, conv = modules[i]
        bn_name, bn = modules[i + 1]

        if not isinstance(conv, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            continue
        if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            warnings.warn(f"fuse_bn_running_var: Could not fuse layer: {conv_name}")
            continue
        assert conv.bias is None

        s = torch.div(1, torch.sqrt(bn.eps + bn.running_var))
        conv.weight.data = torch.einsum('i...,i->i...', conv.weight.data, s)

        if bn.running_mean is not None:
            bn.running_mean.mul_(s)
        bn.running_var.fill_(1.)
        if verbose:
            print(f'fuse_bn_running_var: process {bn_name}')

    return model
