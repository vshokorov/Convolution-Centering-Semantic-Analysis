import torch
import torch.nn as nn


class BiasWeightChangeBatchNormNothing(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        # x: [B, C, T]
        if self.training:
            var = x.var(dim=[0, 2], unbiased=False)
            with torch.no_grad():
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            var = self.running_var
        
        x_norm = x / torch.sqrt(var.view(1, -1, 1) + self.eps)
        return self.weight.view(1, -1, 1) * x_norm
    
    
class BiasWeightChangeBatchNormOnlyMean(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        # x: [B, C, T]
        if self.training:
            mean = x.mean(dim=[0, 2])
            var = x.var(dim=[0, 2], unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        return self.weight.view(1, -1, 1) * x_norm
    
    
class BiasWeightChangeBatchNormOnlyBias(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        # x: [B, C, T]
        if self.training:
            var = x.var(dim=[0, 2], unbiased=False)
            with torch.no_grad():
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            var = self.running_var
            
        x_norm = x / torch.sqrt(var.view(1, -1, 1) + self.eps)
        return self.weight.view(1, -1, 1) * (x_norm + self.bias.view(1, -1, 1))
    
    
class BiasWeightChangeBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        # x: [B, C, T]
        if self.training:
            mean = x.mean(dim=[0, 2])
            var = x.var(dim=[0, 2], unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        return self.weight.view(1, -1, 1) * (x_norm + self.bias.view(1, -1, 1))

    
def replace_batchnorm_with_norm(model, norm, device):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm1d):
            num_features = module.num_features
            device = module.weight.device
            new_layer = norm(num_features)
            new_layer = new_layer.to(device)
            setattr(model, name, new_layer)
        else:
            replace_batchnorm_with_norm(module, norm, device)
    return model