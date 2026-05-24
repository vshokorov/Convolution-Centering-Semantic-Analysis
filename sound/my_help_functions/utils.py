import torch
import numpy as np

from my_help_functions.calculate_angles_sound import(
    analyze_all_layers,
    analyze_angle_change
)


def get_signal_stamps(convs, bns):
    before_conv = [convs[i][2].detach().cpu() for i in range(len(bns))]
    after_conv = [convs[i][3].detach().cpu() for i in range(len(bns))]
    after_shift = []
    for i in range(len(bns)):
        bn = bns[i][1]
        signal = convs[i][3].detach().cpu()

        if hasattr(bn, 'running_mean') and bn.running_mean is not None:
            signal = signal - bn.running_mean.view(1, -1, 1).detach().cpu()

        if hasattr(bn, 'bias') and bn.bias is not None:
            signal = signal + bn.bias.view(1, -1, 1).detach().cpu()

        after_shift.append(signal)
    return before_conv, after_conv, after_shift


def calculate_and_save_angles_fig6(convs, bns, labels, save_dir, norm_type_name):
    
    before_conv, after_conv, after_shift = get_signal_stamps(convs, bns)
    
    label_names = asr_model.cfg.labels + ['silence']

    intra_class_sims, inter_class_sims, class_vs_noise_sims = analyze_all_layers(
        after_conv, 
        labels, 
        label_names
    )
    
    results_dict = {
        'intra_class_sims': intra_class_sims,
        'inter_class_sims': inter_class_sims,
        'class_vs_noise_sims': class_vs_noise_sims
    }

    torch.save(results_dict, f'{save_dir}/{norm_type_name}_after_conv.pt')
    
    intra_class_sims, inter_class_sims, class_vs_noise_sims = analyze_all_layers(
        after_shift, 
        labels, 
        label_names
    )
    
    results_dict = {
        'intra_class_sims': intra_class_sims,
        'inter_class_sims': inter_class_sims,
        'class_vs_noise_sims': class_vs_noise_sims
    }

    torch.save(results_dict, f'{save_dir}/{norm_type_name}_after_shift.pt')
    
    intra_class_sims, inter_class_sims, class_vs_noise_sims = analyze_angle_change(
        before_conv, after_conv,
        labels, 
        label_names
    )

    results_dict = {
        'intra_class_sims': intra_class_sims,
        'inter_class_sims': inter_class_sims,
        'class_vs_noise_sims': class_vs_noise_sims,
    }

    torch.save(results_dict, f'{save_dir}/{norm_type_name}_change_conv.pt')
    
    intra_class_sims, inter_class_sims, class_vs_noise_sims = analyze_angle_change(
        after_conv, after_shift,
        labels, 
        label_names
    )

    results_dict = {
        'intra_class_sims': intra_class_sims,
        'inter_class_sims': inter_class_sims,
        'class_vs_noise_sims': class_vs_noise_sims,
    }

    torch.save(results_dict, f'{save_dir}/{norm_type_name}_change_shift.pt')
    

def calculate_and_save_angles_fig8(asr_model, convs, labels, res, corr_type):
    before_conv = [convs[k][0] for k in range(len(convs))]
    after_conv = [convs[k][1] for k in range(len(convs))]

    label_names = asr_model.cfg.labels + ['silence']
    inside, outside, back = analyze_angle_change(before_conv, after_conv, labels, label_names)
    res[corr_type] = np.mean(np.stack((inside, outside, back)), 0)
    
    return res


def run_validation(model, manifest_path, batch_size=32):    
    extended_labels = model.cfg.labels + ['silence']
    
    model.setup_test_data(
        test_data_config={
            'manifest_filepath': manifest_path,
            'sample_rate': 16000,
            'labels': extended_labels,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 4
        }
    )
    
    model.eval()
    all_preds = []
    all_labels = []
    
    test_dataloader = model.test_dataloader()
    
    with torch.no_grad():
#         for batch in tqdm(test_dataloader):
        batch = next(iter(test_dataloader))
    
        input_signal, input_signal_length, labels, _ = batch

        device = next(model.parameters()).device
        input_signal = input_signal.to(device)
        input_signal_length = input_signal_length.to(device)

        logits = model.forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length
        )

        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())    
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    correct = np.sum(all_preds == all_labels)
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0
    
    label_stats = {}
    for label_idx in range(len(extended_labels)):
        label_name = extended_labels[label_idx]
        mask = all_labels == label_idx
        if np.sum(mask) > 0:
            label_correct = np.sum((all_preds == all_labels) & mask)
            label_total = np.sum(mask)
            label_stats[label_name] = {
                'correct': int(label_correct),
                'total': int(label_total),
                'accuracy': label_correct / label_total
            }
    
    return all_labels, all_preds, accuracy