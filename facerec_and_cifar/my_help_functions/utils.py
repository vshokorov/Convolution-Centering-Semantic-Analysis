import torch
import torch.nn.functional as F

__all__ = ['get_sampled_cos', 'get_sampled_cos_paired', 'ScalableMask']

def get_sampled_cos(a_indices, b_indices, features, features2=None, reduce=None, fix_order=False, n_samples=10000):
    if b_indices is None:
        if features2 is not None:
            assert features2.ndim == 1
        assert fix_order == False
        return get_sampled_cos_1d(a_indices, features, features2, reduce, n_samples)
    else:
        return get_sampled_cos_2d(a_indices, b_indices, features, features2, reduce, fix_order, n_samples)


def get_sampled_cos_3d(a_indices, b_indices, features, features2=None, reduce=None, fix_order=False, n_samples=1000):
    '''
    Sampled cos for full sample features.
    '''
    assert not fix_order
    
    _, q = torch.unique(a_indices[:, 0], return_counts=True)
    INDICES_PER_SAMPLE = q[0].item()
    A_SAMPLES_NUMBER = q.size(0)
    assert torch.all(q == INDICES_PER_SAMPLE)

    _, q = torch.unique(b_indices[:, 0], return_counts=True)
    B_SAMPLES_NUMBER = q.size(0)
    assert torch.all(q == INDICES_PER_SAMPLE)
    b_indices = b_indices.view(B_SAMPLES_NUMBER, INDICES_PER_SAMPLE, 3)


    if n_samples > A_SAMPLES_NUMBER * B_SAMPLES_NUMBER / 10:
        n_samples = int(A_SAMPLES_NUMBER * B_SAMPLES_NUMBER / 10)
        if n_samples < 10:
            return None

    a_selected_indices_idx = torch.linspace(0, A_SAMPLES_NUMBER-1, n_samples, dtype=int, device=features.device)
    b_selected_indices_idx = torch.randint(0, B_SAMPLES_NUMBER, (n_samples,), device=features.device)

    a_selected_indices = a_indices.view(A_SAMPLES_NUMBER, INDICES_PER_SAMPLE, 3)[a_selected_indices_idx].reshape(-1, 3)
    b_selected_indices = b_indices.view(B_SAMPLES_NUMBER, INDICES_PER_SAMPLE, 3)[b_selected_indices_idx].reshape(-1, 3)

    def get_cos_for_features(f):
        a = f[a_selected_indices[:, 0], :, a_selected_indices[:, 1], a_selected_indices[:, 2]].reshape(n_samples, -1)
        b = f[b_selected_indices[:, 0], :, b_selected_indices[:, 1], b_selected_indices[:, 2]].reshape(n_samples, -1)
        return torch.sum(
            F.normalize(a, dim=1) * F.normalize(b, dim=1),
            dim=1
        )
    cos = get_cos_for_features(features)
    if not features2 is None:
        cos = get_cos_for_features(features2) - cos
    return reduce_cos_stats(cos, reduce)

def get_sampled_cos_2d(a_indices, b_indices, features, features2=None, reduce=None, fix_order=False, n_samples=10000):
    '''
    Sampled cos for feature maps.
    '''
    if n_samples > a_indices.size(0) * b_indices.size(0) / 10:
        n_samples = int(a_indices.size(0) * b_indices.size(0) / 10)
        if n_samples < 10:
            return None
    
    a_selected_indices_idx = torch.linspace(0, a_indices.size(0)-1, n_samples, dtype=int, device=features.device)

    if fix_order:
        assert a_indices.size(0) == b_indices.size(0)
        b_selected_indices_idx = a_selected_indices_idx
    else:
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
    return reduce_cos_stats(cos, reduce)


def get_sampled_cos_1d(indices, features, vector2=None, reduce=None, n_samples=10000):
    '''
    Sampled cos for feature map vs vector
    '''
    if indices.size(0) < 10:
        return None

    if indices.size(0) > n_samples:
        a_selected_indices_idx = torch.linspace(0, indices.size(0)-1, n_samples, dtype=int, device=features.device)
        a_selected_indices = indices[a_selected_indices_idx]
        features = features[a_selected_indices[:, 0], :, a_selected_indices[:, 1], a_selected_indices[:, 2]]
    else:
        features = features[indices[:, 0], :, indices[:, 1], indices[:, 2]]
    
    if vector2 is None:
        assert features.size(1) == 1
        cos = features.flatten()
    else:
        cos = torch.sum(
            F.normalize(features, dim=1) * F.normalize(vector2, dim=0)[None],
            dim=1
        ).flatten()
    
    return reduce_cos_stats(cos, reduce)



def reduce_cos_stats(cos, reduce, dim=None):
    if reduce is None:
        return cos
    elif reduce == 'mean':
        return {'mean': cos.mean(dim=dim)}
    elif reduce == 'var_mean':
        return {k: v for k, v in zip(['var', 'mean'], torch.var_mean(cos, dim=dim))}
    elif reduce == 'quantiles':
        if len(coco_dl) > 1:
            print('WARNING: quantile method is not suitable for batching!')
        qs = torch.Tensor([0.05, 0.25, 0.5, 0.75, 0.95]).to(cos.device)
        return {f'q{q.item():.2f}': v.item() for q, v in zip(qs, torch.quantile(cos, qs, dim=dim))}
    else:
        raise


class ScalableMask: 
    def __init__(self, mask_size=(1, 320, 320), padding_p=0.1, device='cpu'):
        assert len(mask_size) == 3
        self.orig_mask = torch.zeros(
            mask_size, 
            dtype=torch.bool, device=device
        )
        pad_dx = int(mask_size[2] * padding_p)
        pad_dy = int(mask_size[1] * padding_p)
        self.orig_mask[:, pad_dy:mask_size[1]-pad_dy, pad_dx:mask_size[2]-pad_dx] = 1

        self.old_size = mask_size
        self.mask = self.orig_mask
        torch.cuda.empty_cache()
    
    def _create_mask(self, size):
        del self.mask

        self.mask = F.interpolate(self.orig_mask[None].half(), size=size, mode='nearest')[0] > 0
        torch.cuda.empty_cache()

        self.old_size = size
    
    def create_mask(self, size):
        if size[0] != self.old_size[0] or size[1] != self.old_size[1]:
            self._create_mask(size)

        return self.mask






def get_sampled_cos_paired(issame, a_indices, b_indices, features, features2=None, reduce=None, fix_order=False, n_samples=1000):
    '''
    Sampled cos for feature maps paired!
    '''
    if n_samples > a_indices.size(0) * b_indices.size(0) / 10:
        n_samples = int(a_indices.size(0) * b_indices.size(0) / 10)
        if n_samples < 4:
            return None
    
    assert features.size(0) % 2 == 0
    assert torch.isclose(a_indices[:, 0], torch.tensor(0)).all()
    assert torch.isclose(b_indices[:, 0], torch.tensor(1)).all()
    n_pairs = features.size(0) // 2
    feature_size = features.size(2) * features.size(3)
    batch_shift_vector = torch.arange(n_pairs, device=features.device).mul(2).repeat_interleave(n_samples)
    
    if a_indices.size(0) > b_indices.size(0):
        a_selected_indices_idx = torch.linspace(0, a_indices.size(0)-1, n_samples, dtype=int, device=features.device)
        b_selected_indices_idx = torch.randint(0, b_indices.size(0), (n_samples,), device=features.device)
    else:
        a_selected_indices_idx = torch.randint(0, a_indices.size(0), (n_samples,), device=features.device)
        b_selected_indices_idx = torch.linspace(0, b_indices.size(0)-1, n_samples, dtype=int, device=features.device)

    a_selected_indices = a_indices[a_selected_indices_idx].repeat(n_pairs, 1)
    a_selected_indices[:, 0] += batch_shift_vector
    b_selected_indices = b_indices[a_selected_indices_idx if fix_order else b_selected_indices_idx].repeat(n_pairs, 1)
    b_selected_indices[:, 0] += batch_shift_vector


    def get_cos_for_features(f):
        a = f[a_selected_indices[:, 0], :, a_selected_indices[:, 1], a_selected_indices[:, 2]]
        b = f[b_selected_indices[:, 0], :, b_selected_indices[:, 1], b_selected_indices[:, 2]]
        return torch.sum(
            F.normalize(a, dim=1, eps=1e-7) * F.normalize(b, dim=1, eps=1e-7),
            dim=1
        )
    cos = get_cos_for_features(features)
    if not features2 is None:
        cos = get_cos_for_features(features2) - cos
    
    cos = cos.view(n_pairs, n_samples)
    return {
        'pos': reduce_cos_stats(cos[issame], reduce),
        'neg': reduce_cos_stats(cos[~issame], reduce),
    }