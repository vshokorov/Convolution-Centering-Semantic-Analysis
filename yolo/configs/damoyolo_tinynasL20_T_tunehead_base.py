#!/usr/bin/env python3

import os

from configs.damoyolo_tinynasL20_T import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.num_workers = 8
        # optimizer
        self.train.batch_size = 160
        self.train.base_lr_per_img = 0.005 / 64
        self.train.no_aug_epochs = 2
        self.train.warmup_epochs = 1
        self.train.total_epochs = 6

        self.train.finetune_path = './damoyolo_tinynasL20_T_420.pth'
        self.train.load_head_weights = True
        self.train.freeze_param_name_pattern = r'^(?!(head)).*'
        self.train.freeze_bn_name_pattern = r'.*'
        self.train.ema_momentum = 0.998

        self.model.head['use_preclass_BN'] = False
        self.model.head['use_coslogit'] = False



