#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import torch
from torch.nn import functional
from loguru import logger
from tqdm import tqdm

from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from damo.dataset import build_dataloader, build_dataset
from damo.detectors.detector import build_ddp_model, build_local_model
from damo.utils import fuse_model, get_model_info, setup_logger, synchronize
from damo.utils import all_gather, get_world_size, is_main_process, synchronize
from damo.utils.timer import Timer, get_time_str



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_on_dataset(model, data_loader, device, conv_result_list, input_layer_process_func, timer=None, tta=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device('cpu')
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
                output = model(images.to(device))

            if timer:
                # torch.cuda.synchronize() # consume much time
                timer.toc()
            
            conv_input = conv_result_list.pop()
            output = [input_layer_process_func(f, t) for f, t in zip(conv_input, targets)]
            
        results_dict.update(
            {img_id: result
             for img_id, result in zip(image_ids, output)})
    return results_dict

def inference(
    model,
    data_loader,
    dataset_name,
    conv_result_list,
    input_layer_process_func,
    device='cuda',
    multi_gpu_infer=True,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    dataset = data_loader.dataset
    logger.info('Start evaluation on {} dataset({} images).'.format(
        dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    results = compute_on_dataset(model, data_loader, device,
                                 conv_result_list, input_layer_process_func, inference_timer)
    # wait for all processes to complete before measuring the time
    if multi_gpu_infer:
        synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        'Total run time: {} ({} s / img per device, on {} devices)'.format(
            total_time_str, total_time * num_devices / len(dataset),
            num_devices))
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        'Model inference time: {} ({} s / img per device, on {} devices)'.
        format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        ))

    return results
    


def make_parser():
    parser = argparse.ArgumentParser('damo eval')

    # distributed
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument(
        '-f',
        '--config_file',
        default=None,
        type=str,
        help='pls input your config file',
    )
    parser.add_argument(
        '--dataset',
        dest='dataset',
        choices=['train', 'val'],
        help='Evaluating on train/val set.',
    )
    parser.add_argument(
        '--extractor',
        dest='extractor',
        choices=['9pts_per_bbox', '25pts_per_image'],
        help='How to extract emmbeding for sample',
    )
    parser.add_argument(
        '--hook_location',
        dest='hook_location',
        choices=['inner', 'preclass', 'preBNclass'],
    )
    parser.add_argument('-c',
                        '--ckpt',
                        default=None,
                        type=str,
                        help='ckpt for eval')
    parser.add_argument('--conf', default=None, type=float, help='test conf')
    parser.add_argument('--nms',
                        default=None,
                        type=float,
                        help='test nms threshold')
    parser.add_argument('--tsize',
                        default=None,
                        type=int,
                        help='test img size')
    parser.add_argument('--seed', default=None, type=int, help='eval seed')
    parser.add_argument(
        '--fuse',
        dest='fuse',
        default=False,
        action='store_true',
        help='Fuse conv and bn for testing.',
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

class EmbsExtractor():
    w = None

    def __embs_to_3x3(self, sample_embs):
        if self.w is None:
            nf = sample_embs.size(0)
            w = torch.zeros(nf*9, nf, 3, 3)
            for i in range(3):
                for j in range(3):
                    w[(3*i + j)*nf:(3*i + j + 1)*nf, :, i, j] = torch.eye(nf)
            self.w = w.to(sample_embs.device)
        
        return functional.conv2d(sample_embs.unsqueeze(0), self.w, stride=(1, 1), padding=(1, 1)).squeeze()


    def get_embs_9pts_per_bbox(self, sample_embs, target):
        cs = (target.bbox[:, :2] + target.bbox[:, 2:]) / 2
        points = torch.stack([
            cs, 
            (cs + target.bbox[:, [0, 1]]) / 2, 
            (2*cs + target.bbox[:, [0, 1]] + target.bbox[:, [0, 3]]) / 4, 
            (cs + target.bbox[:, [0, 3]]) / 2, 
            (2*cs + target.bbox[:, [0, 3]] + target.bbox[:, [2, 1]]) / 4,
            (cs + target.bbox[:, [2, 1]]) / 2, 
            (2*cs + target.bbox[:, [2, 1]] + target.bbox[:, [2, 3]]) / 4,
            (cs + target.bbox[:, [2, 3]]) / 2,
            (2*cs + target.bbox[:, [2, 3]] + target.bbox[:, [0, 1]]) / 4,
        ]).permute(1, 0, 2).reshape(-1, 2)
        assert target.size[0] == target.size[1]
        assert sample_embs.size(1) == sample_embs.size(2)
        points *= sample_embs.size(1) / target.size[0]
        points = points.long().to(sample_embs.device)

        expanded_sample_embs = self.__embs_to_3x3(sample_embs)
        local_embs = expanded_sample_embs[:, points[:, 0], points[:, 1]].cpu()
        local_embs = local_embs.reshape(3, 3, sample_embs.size(0), len(points)).permute(3, 2, 0, 1)
        labels = target.extra_fields['labels'].clone().repeat_interleave(9)
        return {'embs': local_embs, 'labels': labels}

    def get_embs_25pts_per_image(self, sample_embs, target):
        assert sample_embs.size(1) == sample_embs.size(2)
        points = torch.linspace(0, sample_embs.size(1), 7, dtype=int)[1:-1]
        points = torch.stack(torch.meshgrid(points, points))\
            .permute(2, 1, 0)\
            .reshape(-1, 2)\
            .to(sample_embs.device)

        expanded_sample_embs = self.__embs_to_3x3(sample_embs)
        local_embs = expanded_sample_embs[:, points[:, 0], points[:, 1]].cpu()
        return {'embs': local_embs}

class COCOSubset(torch.utils.data.Subset):
    def __setattr__(self, key, value):
        if key in ['dataset', 'indices']:
            if hasattr(self, key):
                print('WARNING: COCOSubset try to set new', key)
            else:
                super().__setattr__(key, value)
        else:
            self.dataset.__setattr__(key, value)

@logger.catch
def main():
    args = make_parser().parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    synchronize()

    device = 'cuda'
    config = parse_config(args.config_file)
    config.merge(args.opts)

    save_dir = os.path.join(config.miscs.output_dir, config.miscs.exp_name)

    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    setup_logger(save_dir,
                 distributed_rank=args.local_rank,
                 mode='w')
    logger.info('Args: {}'.format(args))

    model = build_local_model(config, device)
    model.head.nms = True

    model.cuda(args.local_rank)
    model.eval()

    ckpt_file = args.ckpt
    logger.info('loading checkpoint from {}'.format(ckpt_file))
    loc = 'cuda:{}'.format(args.local_rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    new_state_dict = {}
    for k, v in ckpt['model'].items():
        k = k.replace('module', '')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    logger.info('loaded checkpoint done.')

    for layer in model.modules():
        if isinstance(layer, RepConv):
            layer.switch_to_deploy()

    infer_shape = sum(config.test.augment.transform.image_max_range) // 2
    logger.info('Model Summary: {}'.format(get_model_info(model,
        (infer_shape, infer_shape))))

    TARGET_CONV_INPUT = []
    if args.hook_location == 'inner':
        hook = model.backbone.block_list[3].block_list[0].conv1.conv1.register_forward_hook(
            lambda m, i, o: TARGET_CONV_INPUT.append(i[0].detach())
        )
    elif args.hook_location == 'preclass':
        hook = model.head.gfl_cls[0].register_forward_hook(
            lambda m, i, o: TARGET_CONV_INPUT.append(i[0].detach())
        )
    elif args.hook_location == 'preBNclass':
        hook = model.neck.merge_5.conv3.bn.register_forward_hook(
            lambda m, i, o: TARGET_CONV_INPUT.append(i[0].detach())
        )
    else:
        raise NotImplementedError()


    embs_extractor = EmbsExtractor()
    if args.extractor == '9pts_per_bbox':
        extractor_func = embs_extractor.get_embs_9pts_per_bbox
    elif args.extractor == '25pts_per_image':
        extractor_func = embs_extractor.get_embs_25pts_per_image
    else:
        raise NotImplementedError()

    model = build_ddp_model(model, local_rank=args.local_rank)
    if args.fuse:
        logger.info('\tFusing model...')
        model = fuse_model(model)

    if args.dataset == 'train':
        full_dataset = build_dataset(config, config.dataset.train_ann, is_train=False)
        dataset = [
            COCOSubset(
                d, list(range(0, len(d), len(d)//5000)))
            for d in full_dataset
        ]
        
        dataset_ann = config.dataset.train_ann
    elif args.dataset == 'val':
        dataset = build_dataset(config, config.dataset.val_ann, is_train=False)
        dataset_ann = config.dataset.val_ann
    else:
        raise NotImplementedError()
    
    # start evaluate
    output_folders = [None] * len(dataset_ann)

    if args.local_rank == 0 and config.miscs.output_dir:
        for idx, dataset_name in enumerate(dataset_ann):
            output_folder = os.path.join(config.miscs.output_dir, 'inference',
                                         dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    
    loader = build_dataloader(dataset,
                              config.test.augment,
                              batch_size=config.test.batch_size,
                              num_workers=config.miscs.num_workers,
                              is_train=False,
                              size_div=32)

    for output_folder, dataset_name, data_loader in zip(
            output_folders, dataset_ann, loader):
        results = inference(
            model,
            data_loader,
            dataset_name,
            TARGET_CONV_INPUT,
            extractor_func,
            device=device,  
        )
    
        torch.save(
            results, 
            os.path.join(output_folder, f'{args.hook_location}_{args.extractor}.pt')
        )


if __name__ == '__main__':
    main()
