# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from conf import settings
    from utils import get_network, get_training_dataset, get_test_dataset, WarmUpLR, \
        most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, init_logging
except:
    from .conf import settings
    from .utils import get_network, get_training_dataset, get_test_dataset, WarmUpLR, \
        most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, init_logging
    

class FacenDataset(Dataset):
    def __init__(self, orig_ds, face_weight, test=False):
        self.faces = nn.functional.interpolate(
            torch.load(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '../images_top100classes.pt'), 
                map_location='cpu'), 
            size=(32, 32)
        )
        self.faces.sub_(torch.Tensor([0.0553, -0.1511, -0.2519]).view(1, 3, 1, 1))
        self.faces.div_(torch.Tensor([0.2508, 0.1737, 0.1459]).view(1, 3, 1, 1))
        self.test_face = self.faces.mean(0)

        self.orig_ds = orig_ds
        assert face_weight >= 0
        self.face_weight = face_weight
        self.test = test
        
    def __len__(self):
        return len(self.orig_ds)
        
    def __getitem__(self, idx):
        image, label = self.orig_ds[idx]

        if self.test:
            image.add_(self.test_face, alpha=self.face_weight)
        else:
            fid = np.random.randint(self.faces.size(0))
            image.add_(self.faces[fid], alpha=self.face_weight)

        return image, label

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        if writer:
            last_layer = list(net.children())[-1]
            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        sys.stdout.write('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))
        sys.stdout.write('\r')
        sys.stdout.flush()

        #update training loss for each iteration
        if writer: 
            writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()
    sys.stdout.write('\n')
    logging.info("{{'train_loss': {}, 'lr': {}, 'epoch': {}}}".format(
        loss.item(), optimizer.param_groups[0]['lr'], epoch
    ))

    if writer:
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    # if args.gpu:
    #     print('GPU INFO.....')
    #     print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    logging.info("{{'test_loss': {}, 'test_acc': {}, 'epoch': {}}}".format(
        test_loss / len(cifar100_test_loader.dataset), 
        correct.float() / len(cifar100_test_loader.dataset), 
        epoch))
    print()

    #add informations to tensorboard
    if tb and writer:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


cifar100_training_dataset = get_training_dataset(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
)

cifar100_test_dataset = get_test_dataset(
    settings.CIFAR100_TRAIN_MEAN,
    settings.CIFAR100_TRAIN_STD,
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-face_weight', type=float, default=0.1)
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

    #data preprocessing:
    
    cifar100_training_loader = DataLoader(
        FacenDataset(cifar100_training_dataset, args.face_weight), shuffle=True, num_workers=4, batch_size=args.b)
    cifar100_test_loader = DataLoader(
        FacenDataset(cifar100_test_dataset, args.face_weight, True), shuffle=True, num_workers=4, batch_size=args.b)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if settings.LOG_DIR:
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)

        #since tensorboard can't overwrite old values
        #so the only way is to create a new tensorboard log
        writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.net, settings.TIME_NOW))
        input_tensor = torch.Tensor(1, 3, 32, 32)
        if args.gpu:
            input_tensor = input_tensor.cuda()
        writer.add_graph(net, input_tensor)
    else:
        writer = None

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    init_logging(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    logging.info(f'Run with args: {args}')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            # print('saving weights file to {}'.format(weights_path))
            # torch.save(net.state_dict(), weights_path)
            best_acc = acc
            # continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    if writer:
        writer.close()
