# import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib as mpl
# %matplotlib inline
# import seaborn as sns
# from tqdm import tqdm

# from sklearn.metrics.pairwise import cosine_similarity

# from damo.config.base import parse_config
# from damo.detectors.detector import build_local_model
# from damo.dataset import build_dataloader
# from tools.demo import Infer

# from my_help_functions.hooks import register_hooks
# from my_help_functions.datasets import CustomCocoDataset, get_CocoNonShuffleDataLoader
# from my_help_functions.visualise_arch import ArchVisualiser
# from my_help_functions.figure1 import BatchT, get_sampled_cos, plot_S_arrow

# from damo.utils import fuse_model, fuse_bn_running_var, get_model_info, setup_logger, synchronize
# from collections import defaultdict, Counter


class BatchT: 
    def __init__(self, batch, device, config, init_mask=True):
        torch.cuda.empty_cache()
        self.old_size = (-1, -1)
        self.mask = None
        self.back_mask = None
        
        self.device = device
        self.config = config

        self.imgs = batch[0].to(self.device)
        self.targets = batch[1]
        self.img_ids = batch[2]
        self.unique_labels = set()
        
        self.orig_mask = np.zeros(
            (len(self.config.dataset.class_names), len(self.img_ids), 320, 320), 
            dtype=np.uint8
        )
        self.orig_back_mask = torch.zeros(
            (len(self.img_ids), 320, 320), 
            dtype=torch.bool, device='cpu'
        )
        assert self.orig_back_mask.size(1) == self.orig_back_mask.size(2) == 320
        self.orig_back_mask[:, 32:-32, 32:-32] = 1

        for i, t, in enumerate(self.targets):
            t = t.resize((320, 320))
            
            for bb, l, seg_mask in zip(t.bbox.long(), t.get_field('labels'), t.get_field('seg_masks').contours):
                if init_mask:
                    mask = np.zeros(t.size, dtype=np.uint8)
                    cv2.fillPoly(mask, pts = [seg_mask.astype(np.int32)], color=(1,))
                    # self.orig_mask[l.item()][i][bb[1]:bb[3], bb[0]:bb[2]] = 1
                    self.orig_mask[l.item()][i] = mask

                self.orig_back_mask[i, bb[1]:bb[3], bb[0]:bb[2]] = 0
                self.unique_labels.add(l.item())

        self.orig_mask = torch.from_numpy(self.orig_mask).to(device) > 0
        self.orig_back_mask = self.orig_back_mask.to(device)
        
        torch.cuda.empty_cache()
    
    def init_mask_by_pos_inds(self, pos_inds, pos_labels):
        self.orig_mask.fill_(0)
        self.old_size = (-1, -1)
        self.pos_at_scale = [{80: 0, 40: 0, 20: 0} for _ in range(len(self))]

        for j in range(len(self)):
            for i in range(3):
                if i == 0:
                    inds_mask = torch.logical_and(j * 8400 <= pos_inds, pos_inds < j * 8400 + 6400)
                    shift, scale = j * 8400, 80
                elif i == 1:
                    inds_mask = torch.logical_and(j * 8400 + 6400 <= pos_inds, pos_inds < j * 8400 + 8000)
                    shift, scale = j * 8400 + 6400, 40
                else:
                    inds_mask = torch.logical_and(j * 8400 + 8000 <= pos_inds, pos_inds < j * 8400 + 8400)
                    shift, scale = j * 8400 + 8000, 20
                
                a = pos_inds[inds_mask]
                a -= shift
                for l in torch.unique(pos_labels[inds_mask]):
                    mask = torch.zeros(scale, scale, dtype=bool, device=self.device)
                    mask.flatten()[a[pos_labels[inds_mask] == l].cpu()] = 1

                    self.orig_mask[l, j].add_(
                        F.interpolate(
                            mask[None, None].half(), 
                            size=(self.orig_mask.size(2), self.orig_mask.size(3)), 
                            mode='nearest'
                        )[0, 0] > 0
                    )
                    self.pos_at_scale[j][scale] += mask.sum().item()
        self.pos_at_scale = [max(x.items(), key=lambda i: i[1])[0] for x in self.pos_at_scale]
        
    
    def _create_mask(self, size):
        del self.mask
        del self.back_mask

        self.mask = torch.zeros(
            (self.orig_mask.size(0), self.orig_mask.size(1), size[0], size[1]), dtype=torch.bool, device=self.device
        )
        for i in range(self.orig_mask.size(0)):
            self.mask[i] = F.interpolate(self.orig_mask[i][None].half(), size=size, mode='nearest')[0] > 0
        # self.mask = F.interpolate(self.orig_mask.half(), size=size, mode='nearest') > 0
        self.back_mask = F.interpolate(self.orig_back_mask[None].half(), size=size, mode='nearest')[0] > 0
        torch.cuda.empty_cache()

        self.old_size = size
    
    def create_mask(self, size):
        if size[0] != self.old_size[0] or size[1] != self.old_size[1]:
            self._create_mask(size)

        return self.mask, self.back_mask
    
    def __len__(self):
        return len(self.img_ids)
    

def get_sampled_cos(a, b, ab2_for_diff=None, reduce=None, n_samples=10000):
    if n_samples > a.size(0) * b.size(0) / 10:
        n_samples = int(a.size(0) * b.size(0) / 10)
        if n_samples < 10:
            return None
    
    a_indices = torch.randint(0, a.size(0), (n_samples,), device=a.device)
    b_indices = torch.randint(0, b.size(0), (n_samples,), device=b.device)

    cos = torch.sum(
        F.normalize(a[a_indices], dim=1) * F.normalize(b[b_indices], dim=1),
        # a[a_indices] * b[b_indices],
        dim=1
    )
    if not ab2_for_diff is None:
        cos2 = torch.sum(
            F.normalize(ab2_for_diff[0][a_indices], dim=1) * F.normalize(ab2_for_diff[1][b_indices], dim=1),
            # ab2_for_diff[0][a_indices] * ab2_for_diff[1][b_indices],
            dim=1
        )
        cos.sub_(cos2)

    if reduce is None:
        return cos
    elif reduce == 'mean':
        return cos.mean().item()
    elif reduce == 'var_mean':
        return tuple(map(lambda x: x.item(), torch.var_mean(cos)))
    else:
        raise
        
        
def plot_S_arrow(ax, x1, y1, x2, y2, rad):
    dx, dy = (x2 - x1) / 2, (y2 - y1) / 2
    arrowprops=dict(
        arrowstyle="->", shrinkA=0, shrinkB=2, patchA=None, patchB=None,
        # connectionstyle=f'arc3,rad={rad}',
        connectionstyle=f'angle3,angleA=0,angleB={rad}',
    )
    for _ in range(2):
        ax.annotate("",
                    xy=(x1, y1), xycoords='data',
                    xytext=(x1+dx, y1+dy), textcoords='data',
                    arrowprops=arrowprops
                    )
        x1, y1 = x1+dx, y1+dy
        # arrowprops['connectionstyle'] = f'arc3,rad={-rad}'
        arrowprops['connectionstyle'] = f'angle3,angleA={rad},angleB=0'
        arrowprops['arrowstyle'] = '-'
        arrowprops['shrinkA'] = 5
        arrowprops['shrinkB'] = 0