# from torchvision.datasets.coco import CocoDetection
from damo.dataset import build_dataloader, build_dataset
from damo.dataset.collate_batch import BatchCollator
from damo.dataset.transforms import build_transforms
from damo.structures.bounding_box import Segmentation_extra_field
import torchvision
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, Sampler


# dataset = build_dataset(config, config.dataset.train_ann, is_train=False)
# assert(len(dataset) == 1)
# dataset = dataset[0]

# image_sizes = {}
# for img_id in tqdm(range(len(dataset))):
#     image, target, _ = dataset[img_id]
#     image_sizes[img_id] = image.shape

# c_ds = Counter()
# c    = Counter()
# images_list = []
# empty_images_list = []
# for img_id in tqdm(range(len(dataset))):
#     img_size = image_sizes[img_id][:2][::-1]
#     anno = dataset._load_target(dataset.ids[img_id])
#     anno = [obj for obj in anno if obj['iscrowd'] == 0]

#     boxes = [obj['bbox'] for obj in anno]
#     boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
#     target = BoxList(boxes, img_size, mode='xywh').convert('xyxy')
#     target = target.clip_to_image(remove_empty=True)

#     classes = [obj['category_id'] for obj in anno]
#     classes = [dataset.contiguous_class2id[dataset.ori_id2class[c]] 
#                 for c in classes]
#     c_ds.update(classes)

#     skip_image = False
#     for target_id1, bb1 in enumerate(target.bbox):
#         for target_id2, bb2 in enumerate(target.bbox):
#             if target_id2 <= target_id1:
#                 continue

#             if (max(bb1[2], bb2[2]) - min(bb1[0], bb2[0]) - (bb1[2] - bb1[0]) - (bb2[2] - bb2[0]) < img_size[1] / 4) and \
#                 (max(bb1[3], bb2[3]) - min(bb1[1], bb2[1]) - (bb1[3] - bb1[1]) - (bb2[3] - bb2[1]) < img_size[0] / 4):
#                 skip_image = True
#                 break
        
#         if skip_image: break
    
#     if not skip_image: 
#         if len(target) == 0:
#             empty_images_list.append(img_id)
#         else:
#             if min(class_probs[c] for c in classes) > np.random.rand():    
#                 images_list.append(img_id)
#                 c.update(classes)

# class_probs = {}
# for i in c_ds.keys():
#     class_probs[i] = min(1, (c_ds[i] / sum(c_ds.values())) / (c[i] / sum(c.values())))

# plt.plot([i[1]    / sum(c_ds.values()) for i in c_ds.most_common()])
# plt.plot([c[i[0]] / sum(c.values())    for i in c_ds.most_common()])

class CustomCocoDataset():
    def __init__(self, cfg, load_train=True):
        ann_files = cfg.dataset.train_ann if load_train else cfg.dataset.val_ann

        dataset = build_dataset(cfg, ann_files, is_train=False)
        assert(len(dataset) == 1)
        self._dataset = dataset[0]

        with open('dataset_images_list', 'rb') as fr:
            dataset_infos = pickle.load(fr)
        
        if load_train:
            self.images_list = dataset_infos['train']
            self.empty_images_list = dataset_infos['train_empty']
        else:
            self.images_list = dataset_infos['val']
            self.empty_images_list = dataset_infos['val_empty']

        self._class_to_images = {}
        for ds_img_id, image_name in enumerate(self.images_list):
            anno = self._dataset._load_target(self._dataset.ids[image_name])
            anno = [obj for obj in anno if obj['iscrowd'] == 0]
            classes = [obj['category_id'] for obj in anno]
            classes = [self._dataset.contiguous_class2id[self._dataset.ori_id2class[c]] 
                        for c in classes]
            for c in classes:
                if not c in self._class_to_images:
                    self._class_to_images[c] = []
                self._class_to_images[c].append(ds_img_id)

        transforms = cfg.test.augment.transform
        transforms = build_transforms(
            start_epoch = 0,
            total_epochs = 1,
            no_aug_epochs = 0,
            iters_per_epoch = 1,
            num_workers = cfg.miscs.num_workers,
            batch_size = 1,
            num_gpus = 1,
            **transforms
        )
        self._dataset._transforms = transforms
        if hasattr(self._dataset, '_dataset'):
            self._dataset._dataset._transforms = transforms
    
    def __getitem__(self, idx, return_empty=False):
        if return_empty:
            idx = self.empty_images_list[idx]
        else:
            idx = self.images_list[idx]

        img, target, seg_masks, idx = self._dataset.pull_item(idx, return_as_BoxList=True)
        ef = Segmentation_extra_field(seg_masks, (img.shape[1], img.shape[0]))
        target.add_field('seg_masks', ef)

        if self._dataset._transforms is not None:
            img, target = self._dataset._transforms(img, target)
        return img, target, idx

    def __len__(self) -> int:
        return len(self.images_list)

    def get_ids_for_class(self, label):
        return self._class_to_images[label]

    def get_classes(self):
        return list(self._class_to_images.keys())

class CustomCocoBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        classes = self.dataset.get_classes()
        for c in classes:
            if (l := len(self.dataset.get_ids_for_class(c))) < batch_size:
                print(f'WARNING: class {c} have only {l} samples')

    def __iter__(self):
        classes = self.dataset.get_classes()
        for c in classes:
            yield self.dataset.get_ids_for_class(c)[:self.batch_size]

    def __len__(self):
        return len(self.dataset.get_classes())
    

def get_CocoNonShuffleDataLoader(cfg, dataset=None, batch_size=None):
    if dataset is None:
        dataset = CustomCocoDataset(cfg)
    
    if batch_size is None:
        batch_size = cfg.train.batch_size

    batch_sampler = CustomCocoBatchSampler(dataset, batch_size)
    collator = BatchCollator(size_divisible=32)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )
    return data_loader
