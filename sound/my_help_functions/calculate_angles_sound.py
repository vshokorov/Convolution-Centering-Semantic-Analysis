import torch
import numpy as np
from tqdm.auto import tqdm

def calculate_layer_similarities(feature_maps, class_masks, noise_idx, non_noise_indices):
    """
    Calculate similarities for one layer
    
    Args:
        feature_maps: tensor of shape [batch, channels, time]
        class_masks: dict of {class_idx: boolean tensor mask}
        noise_idx: index of noise class
        non_noise_indices: list of non-noise class indices
    
    Returns:
        tuple of (mean, std) for each category:
        - intra_class: (mean, std)
        - inter_class: (mean, std)  
        - class_vs_noise: (mean, std)
    """
    # Separate by class using pre-computed masks
    class_features = {}
    for cls_idx, mask in class_masks.items():
        class_features[cls_idx] = feature_maps[mask]  # [num_samples, channels, time]
    
    # Average over time first, then compute cosine similarity
    # [num_samples, channels, time] -> [num_samples, channels]
    for cls_idx in class_features:
        class_features[cls_idx] = class_features[cls_idx].mean(dim=2)

    # Calculate intra-class similarities
    intra_sims = []
    for cls_idx in class_features.keys():
        feats = class_features[cls_idx]
        num_samples = feats.shape[0]

        if num_samples < 2:
            continue

        # Normalize and compute all pairwise similarities
        feats_norm = torch.nn.functional.normalize(feats, p=2, dim=1)
        sim_matrix = torch.mm(feats_norm, feats_norm.t())
        mask_upper = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        sims = sim_matrix[mask_upper].cpu().numpy()
        intra_sims.extend(sims)

    # Calculate inter-class similarities
    inter_sims = []
    for i, cls1 in enumerate(non_noise_indices):
        if cls1 not in class_features:
            continue
        for cls2 in non_noise_indices[i+1:]:
            if cls2 not in class_features:
                continue

            feats1_norm = torch.nn.functional.normalize(class_features[cls1], p=2, dim=1)
            feats2_norm = torch.nn.functional.normalize(class_features[cls2], p=2, dim=1)
            sim_matrix = torch.mm(feats1_norm, feats2_norm.t())
            inter_sims.extend(sim_matrix.flatten().cpu().numpy())

    # Calculate class-vs-noise similarities
    noise_sims = []
    if noise_idx in class_features:
        noise_feats_norm = torch.nn.functional.normalize(class_features[noise_idx], p=2, dim=1)

        for cls in non_noise_indices:
            if cls not in class_features:
                continue
            cls_feats_norm = torch.nn.functional.normalize(class_features[cls], p=2, dim=1)
            sim_matrix = torch.mm(cls_feats_norm, noise_feats_norm.t())
            noise_sims.extend(sim_matrix.flatten().cpu().numpy())
    
    return (
        (np.mean(intra_sims) if len(intra_sims) > 0 else 0,
         np.std(intra_sims) if len(intra_sims) > 0 else 0),
        (np.mean(inter_sims) if len(inter_sims) > 0 else 0,
         np.std(inter_sims) if len(inter_sims) > 0 else 0),
        (np.mean(noise_sims) if len(noise_sims) > 0 else 0,
         np.std(noise_sims) if len(noise_sims) > 0 else 0)
    )


def analyze_all_layers(feature_maps_list, labels, label_names):
    """
    Analyze all layers and return 3 lists
    
    Args:
        feature_maps_list: list of tensors, each [batch, channels, time]
        labels: array of label indices
        label_names: list of label names
    
    Returns:
        intra_class_list: list of intra-class similarities per layer
        inter_class_list: list of inter-class similarities per layer
        class_vs_noise_list: list of class-vs-noise similarities per layer
    """
    num_layers = len(feature_maps_list)
    
    # Pre-compute masks and indices
    noise_label = 'silence' if 'silence' in label_names else '_silence_'
    noise_idx = list(label_names).index(noise_label)
    
    unique_labels = set(labels)
    non_noise_indices = [idx for idx in unique_labels if idx != noise_idx]
    
    # Pre-compute boolean masks for each class
    class_masks = {}
    for label_idx in unique_labels:
        mask = torch.tensor([l == label_idx for l in labels], dtype=torch.bool)
        class_masks[label_idx] = mask
    
    intra_class_list = []
    inter_class_list = []
    class_vs_noise_list = []
    
    
    for layer_idx in tqdm(range(num_layers)):
        fm = feature_maps_list[layer_idx]
        
        if torch.cuda.is_available():
            fm = fm.cuda()
        
        intra, inter, noise = calculate_layer_similarities(
            fm, class_masks, noise_idx, non_noise_indices
        )
        
        intra_class_list.append(intra)
        inter_class_list.append(inter)
        class_vs_noise_list.append(noise)
    
    return intra_class_list, inter_class_list, class_vs_noise_list


def calculate_paired_differences(before_fm, after_fm, class_masks, noise_idx, non_noise_indices):
    """
    Calculate differences for each pair, then average
    
    Returns:
        ((intra_mean, intra_std), (inter_mean, inter_std), (noise_mean, noise_std))
    """
    # Temporal average
    before_avg = before_fm.mean(dim=2)
    after_avg = after_fm.mean(dim=2)
    
    # Separate by class
    class_features_before = {}
    class_features_after = {}
    for cls_idx, mask in class_masks.items():
        class_features_before[cls_idx] = before_avg[mask]
        class_features_after[cls_idx] = after_avg[mask]
    
    # Intra-class paired differences
    intra_diffs = []
    for cls_idx in class_features_before.keys():
        feats_before = class_features_before[cls_idx]
        feats_after = class_features_after[cls_idx]
        
        if feats_before.shape[0] < 2:
            continue
        
        # Compute similarity matrices
        feats_before_norm = torch.nn.functional.normalize(feats_before, p=2, dim=1)
        feats_after_norm = torch.nn.functional.normalize(feats_after, p=2, dim=1)
        
        sim_before = torch.mm(feats_before_norm, feats_before_norm.t())
        sim_after = torch.mm(feats_after_norm, feats_after_norm.t())
        
        # Compute difference matrix
        diff_matrix = sim_after - sim_before
        
        # Extract upper triangle
        mask_upper = torch.triu(torch.ones_like(diff_matrix), diagonal=1).bool()
        diffs = diff_matrix[mask_upper].cpu().numpy()
        intra_diffs.extend(diffs)
    
    # Inter-class paired differences
    inter_diffs = []
    for i, cls1 in enumerate(non_noise_indices):
        if cls1 not in class_features_before:
            continue
        for cls2 in non_noise_indices[i+1:]:
            if cls2 not in class_features_before:
                continue
            
            # Before
            f1_before_norm = torch.nn.functional.normalize(class_features_before[cls1], p=2, dim=1)
            f2_before_norm = torch.nn.functional.normalize(class_features_before[cls2], p=2, dim=1)
            sim_before = torch.mm(f1_before_norm, f2_before_norm.t())
            
            # After
            f1_after_norm = torch.nn.functional.normalize(class_features_after[cls1], p=2, dim=1)
            f2_after_norm = torch.nn.functional.normalize(class_features_after[cls2], p=2, dim=1)
            sim_after = torch.mm(f1_after_norm, f2_after_norm.t())
            
            # Difference
            diff_matrix = sim_after - sim_before
            inter_diffs.extend(diff_matrix.flatten().cpu().numpy())
    
    # Class-vs-noise paired differences
    noise_diffs = []
    if noise_idx in class_features_before:
        noise_before_norm = torch.nn.functional.normalize(class_features_before[noise_idx], p=2, dim=1)
        noise_after_norm = torch.nn.functional.normalize(class_features_after[noise_idx], p=2, dim=1)
        
        for cls in non_noise_indices:
            if cls not in class_features_before:
                continue
            
            # Before
            cls_before_norm = torch.nn.functional.normalize(class_features_before[cls], p=2, dim=1)
            sim_before = torch.mm(cls_before_norm, noise_before_norm.t())
            
            # After
            cls_after_norm = torch.nn.functional.normalize(class_features_after[cls], p=2, dim=1)
            sim_after = torch.mm(cls_after_norm, noise_after_norm.t())
            
            # Difference
            diff_matrix = sim_after - sim_before
            noise_diffs.extend(diff_matrix.flatten().cpu().numpy())
    
    return (
        (np.mean(intra_diffs) if len(intra_diffs) > 0 else 0,
         np.std(intra_diffs) if len(intra_diffs) > 0 else 0),
        (np.mean(inter_diffs) if len(inter_diffs) > 0 else 0,
         np.std(inter_diffs) if len(inter_diffs) > 0 else 0),
        (np.mean(noise_diffs) if len(noise_diffs) > 0 else 0,
         np.std(noise_diffs) if len(noise_diffs) > 0 else 0)
    )


def analyze_angle_change(before_list, after_list, labels, label_names):
    # Pre-compute masks
    noise_label = 'silence'
    noise_idx = list(label_names).index(noise_label)
    unique_labels = set(labels)
    non_noise_indices = [idx for idx in unique_labels if idx != noise_idx]
    
    class_masks = {}
    for label_idx in unique_labels:
        mask = torch.tensor([l == label_idx for l in labels], dtype=torch.bool)
        class_masks[label_idx] = mask
    
    num_layers = len(before_list)

    intra_class_list = []
    inter_class_list = []
    class_vs_noise_list = []
    
    for layer_idx in tqdm(range(num_layers)):
        before_fm = before_list[layer_idx].cuda() if torch.cuda.is_available() else before_list[layer_idx]
        after_fm = after_list[layer_idx].cuda() if torch.cuda.is_available() else after_list[layer_idx]
        
        intra, inter, noise = calculate_paired_differences(
            before_fm, after_fm, class_masks, noise_idx, non_noise_indices
        )

        intra_class_list.append(intra)
        inter_class_list.append(inter)
        class_vs_noise_list.append(noise)

    return intra_class_list, inter_class_list, class_vs_noise_list