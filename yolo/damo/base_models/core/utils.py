# Copyright (c) OpenMMLab. All rights reserved.

from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn


class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

class Conv2dWithScale(nn.Conv2d):
    def __init__(
            self, 
            n_vectors_for_contour,
            in_channels,
            out_channels,
            kernel_size,
            *args,
            **kwargs
        ):
        super(Conv2dWithScale, self).__init__(
            in_channels, out_channels, kernel_size, *args, **kwargs
        )
        self.center_weight = nn.Parameter(
            torch.zeros((out_channels, 1))
        )

        self.contour_weight = nn.Parameter(
            torch.zeros((out_channels, n_vectors_for_contour))
        )

        self.register_buffer('center_vectors', torch.zeros((out_channels, 1, in_channels)))
        self.register_buffer('contour_vectors', torch.zeros((out_channels, n_vectors_for_contour, in_channels)))

    def forward(self, x):
        assert self.kernel_size[0] == self.kernel_size[1]
        assert self.kernel_size[0] % 2 == 1

        contour_mask = torch.ones(self.kernel_size, dtype=torch.bool, device=self.weight.device)
        contour_mask[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
        w = torch.zeros_like(self.weight)

        center_v = torch.bmm(nn.functional.relu(self.center_weight.unsqueeze(1)), self.center_vectors)
        ccontour_v = torch.bmm(nn.functional.relu(self.contour_weight.unsqueeze(1)), self.contour_vectors)

        w[..., ~contour_mask] = center_v.permute(0, 2, 1)
        w[..., contour_mask] = ccontour_v.permute(0, 2, 1)

        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation)

class Conv2dCos(nn.Conv2d):
    def __init__(
            self, 
            in_channels,
            out_channels,
            kernel_size,
            *args,
            **kwargs
        ):
        super(Conv2dCos, self).__init__(
            in_channels, out_channels, kernel_size, *args, **kwargs
        )

        self.temperature = torch.nn.Parameter(torch.ones(1, dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, x):
        # assert self.kernel_size[0] == self.kernel_size[1] == 1

        w = nn.functional.normalize(self.weight, dim=1) * self.temperature
        x_ = nn.functional.normalize(x, dim=1)

        bias = self.bias.mean(0, keepdim=True).repeat(self.out_channels)

        return nn.functional.conv2d(x_, w, bias, self.stride, self.padding, self.dilation)

class Conv2dFreezed(nn.Conv2d):
    def __init__(
            self, 
            in_channels,
            out_channels,
            kernel_size,
            *args,
            **kwargs
        ):
        super(Conv2dFreezed, self).__init__(
            in_channels, out_channels, kernel_size, *args, **kwargs
        )

        # self.new_weight = torch.nn.Parameter(torch.zeros_like(self.weight))
        self.new_weight = torch.nn.Parameter(torch.zeros_like(self.weight[23]))

    def forward(self, x):
        # w = self.weight.detach() + self.new_weight - self.new_weight.detach()
        
        self.weight.data[23] = 0

        w = self.weight.detach() * 1.0
        w[23] += self.new_weight
        
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation)
    
class FilterBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # positive values in grad - negative classes
        # negative values in grad - positive classes
        return torch.clamp(grad_output, min=0)

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.
    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets
