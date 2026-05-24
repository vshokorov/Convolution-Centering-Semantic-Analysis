import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr

from my_help_functions.norms import replace_batchnorm_with_norm


def switch_to_deploy(model, norm):
    modules = list(model.named_modules())
    for i in range(len(modules) - 1):
        name1, layer1 = modules[i]
        name2, layer2 = modules[i + 1]
        if isinstance(layer1, nn.Conv1d) and isinstance(layer2, norm):
            absorb_bn_var_into_conv(layer1, layer2)

            
def absorb_bn_var_into_conv(conv, bn):
    with torch.no_grad():
        scale = 1.0 / torch.sqrt(bn.running_var + bn.eps)

        conv.weight.data *= scale[:, None, None]
        if conv.bias is not None:
            conv.bias.data *= scale
        
        if hasattr(bn, 'running_mean') and bn.running_mean is not None:
            bn.running_mean.data *= scale

        bn.running_var.fill_(1.0)
        bn.eps = 0.0
        

def prepare_model(ckpt_path, norm, device):
    asr_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="commandrecognition_en_matchboxnet3x2x64_v1")
    asr_model = replace_batchnorm_with_norm(asr_model, norm, device)
    checkpoint = torch.load(ckpt_path, weights_only=False)
    asr_model.load_state_dict(checkpoint['state_dict'])
    switch_to_deploy(asr_model, norm)
    asr_model = asr_model.to(device)
    asr_model.eval()
    return asr_model