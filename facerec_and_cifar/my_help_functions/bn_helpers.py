import torch
from torch import nn
import logging


class CustomBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, use_center=True, use_bias=True,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        if not use_center:
            self.running_mean = None
        if not use_bias:
            self.bias = None

    def forward(self, input):
        self._check_input_dim(input)

        mean = None
        # calculate running estimates
        if self.training:
            
            if self.running_mean is not None:
                mean = input.mean([0, 2, 3])
                with torch.no_grad():
                    self.running_mean = self.momentum * mean\
                        + (1 - self.momentum) * self.running_mean
            
            var = input.var([0, 2, 3], unbiased=False)
            with torch.no_grad():
                self.running_var = self.momentum * var \
                    + (1 - self.momentum) * self.running_var
        else:
            if self.running_mean is not None:
                mean = self.running_mean
            var = self.running_var

        if mean is not None:
            input = input - mean[None, :, None, None]
        input = input * torch.rsqrt(var[None, :, None, None] + self.eps)

        if self.affine:
            if self.bias is not None:
                input += self.bias[None, :, None, None]

            input = input * self.weight[None, :, None, None]

        return input
    
    def __repr__(self):
        s = super().__repr__()
        return s[:-1] + f', use_center={self.running_mean is not None}, use_bias={self.bias is not None})'

class CustomBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, use_center=True, use_bias=True,
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
        if not use_center:
            self.running_mean = None
        if not use_bias:
            self.bias = None

    def forward(self, input):
        self._check_input_dim(input)

        mean = None
        # calculate running estimates
        if self.training:
            
            if self.running_mean is not None:
                mean = input.mean(0)
                with torch.no_grad():
                    self.running_mean = self.momentum * mean\
                        + (1 - self.momentum) * self.running_mean
            
            var = input.var(0, unbiased=False)
            with torch.no_grad():
                self.running_var = self.momentum * var \
                    + (1 - self.momentum) * self.running_var
        else:
            if self.running_mean is not None:
                mean = self.running_mean
            var = self.running_var

        if mean is not None:
            input = input - mean[None, :]
        input = input * torch.rsqrt(var[None, :] + self.eps)

        if self.affine:
            if self.bias is not None:
                input += self.bias[None, :]
            
            input = input * self.weight[None, :]

        return input
    
    def __repr__(self):
        s = super().__repr__()
        return s[:-1] + f', use_center={self.running_mean is not None}, use_bias={self.bias is not None})'


def replace_bn_with(model, target_l):
    assert target_l in ['mean_bias', 'only_mean', 'only_bias', 'nothing']
    
    def replace_one(old_bn, parent, name):
        if isinstance(old_bn, torch.nn.BatchNorm2d):
            CustomBatchNorm = CustomBatchNorm2d
        elif isinstance(old_bn, torch.nn.BatchNorm1d):
            CustomBatchNorm = CustomBatchNorm1d
        else:
            raise
        
        if target_l == 'mean_bias':
            new_m = CustomBatchNorm(old_bn.num_features, use_center=True, use_bias=True, eps=old_bn.eps)
        elif target_l == 'only_mean':
            new_m = CustomBatchNorm(old_bn.num_features, use_center=True, use_bias=False, eps=old_bn.eps)
        elif target_l == 'only_bias':
            new_m = CustomBatchNorm(old_bn.num_features, use_center=False, use_bias=True, eps=old_bn.eps)
        elif target_l == 'nothing':
            new_m = CustomBatchNorm(old_bn.num_features, use_center=False, use_bias=False, eps=old_bn.eps)
        
        if hasattr(new_m, 'weight') and new_m.weight is not None:
            nn.init.constant_(new_m.weight, 1)
        if hasattr(new_m, 'bias') and new_m.bias is not None:
            nn.init.constant_(new_m.bias, 0)

        setattr(parent, name, new_m)

    
    def find_bn_with_parents(module, current_path):
        
        for child_name, child in module.named_children():
            child_path = f"{current_path}.{child_name}" if current_path else child_name

            if isinstance(child, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                replace_one(child, module, child_name)
                logging.info(f'replace {child_path}: {child} -> {getattr(module, child_name)}')
            else:
                find_bn_with_parents(child, child_path)

    find_bn_with_parents(model, '')
