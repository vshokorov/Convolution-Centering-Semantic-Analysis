import torch.nn.functional as F


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

