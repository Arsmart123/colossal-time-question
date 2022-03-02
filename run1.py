#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# 这个train_1文件和train文件其他地方都一样，只是在train中使用的是单层linear_col，而train_1中使用的是linear_col_tensorized
import os
import colossalai
import torchvision
from colossalai.builder import *
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import Accuracy, CrossEntropyLoss, MSELoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.trainer import Trainer
from colossalai.trainer.hooks import (AccuracyHook, LogMemoryByEpochHook,
                                      LogMetricByEpochHook,
                                      LogMetricByStepHook,
                                      LogTimingByEpochHook, LossHook,
                                      LRSchedulerHook, ThroughputHook)
from colossalai.utils import MultiTimer, get_dataloader
from model_zoo.vit import vit_lite_depth7_patch4_32
from torchvision import transforms
from colossalai.nn import Linear1D_Col
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import time
import torch.cuda as cuda

# DATASET_PATH = str(os.environ['DATA'])
#
#
# def build_cifar(batch_size):
#    transform_train = transforms.Compose([
#        transforms.RandomCrop(32, padding=4),
#        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
#        transforms.ToTensor(),
#        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#    ])
#    transform_test = transforms.Compose([
#        transforms.Resize(32),
#        transforms.ToTensor(),
#        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#    ])
#
#    train_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH,
#                                                 train=True,
#                                                 download=True,
#                                                 transform=transform_train)
#    test_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transform_test)
#    train_dataloader = get_dataloader(dataset=train_dataset,
#                                      shuffle=True,
#                                      batch_size=batch_size,
#                                      num_workers=4,
#                                      pin_memory=True)
#    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
#    return train_dataloader, test_dataloader


torch.manual_seed(42)  # 下面的数据就随机生成好了。。。固定一下种子


# torchrun  --nproc_per_node=2   --nnodes=1  --node_rank=0   --master_addr='172.18.126.98'     --master_port='51066'  train_1.py --config='./configs/vit_1d.py'
class trainset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        target = torch.randn(2048 * 3)
        # target = torch.LongTensor(target)
        data = torch.randn(2048)
        # data = torch.FloatTensor(data)
        return data, target

    def __len__(self):
        return 512 * 1


def build_cifar(batch_size):
    train_dataset = trainset()
    test_dataset = trainset()
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      num_workers=0,
                                      pin_memory=True)
    test_dataloader = get_dataloader(dataset=test_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    return train_dataloader, test_dataloader


class MLP_1D(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_1 = Linear1D_Col(in_features=2048, out_features=2048*3, gather_output=True)

    def forward(self, x):
        x = self.linear_1(x)
        # x = self.linear_2(x)
        return x


def train_cifar():
    args = colossalai.get_default_parser().parse_args()
    # standard launch
    # colossalai.launch(config=args.config,
    #                   rank=args.rank,
    #                   world_size=args.world_size,
    #                   local_rank=args.local_rank,
    #                   host=args.host,
    #                   port=args.port)
    # colossalai.launch(config=args.config,
    #              rank=args.rank,
    #              world_size=args.world_size,
    #              local_rank=args.local_rank,
    #              host=args.host,
    #              port=args.port)

    # launch from torchrun
    colossalai.launch_from_torch(config=args.config)

    logger = get_dist_logger()
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    # model = vit_lite_depth7_patch4_32()
    model = MLP_1D()

    train_dataloader, test_dataloader = build_cifar(gpc.config.BATCH_SIZE // gpc.data_parallel_size)
    test_dataloader = None

    # criterion = CrossEntropyLoss(label_smoothing=0.1)
    criterion = MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    steps_per_epoch = len(train_dataloader)

    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS * steps_per_epoch,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS * steps_per_epoch)

    engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model=model,
                                                                                    optimizer=optimizer,
                                                                                    criterion=criterion,
                                                                                    train_dataloader=train_dataloader,
                                                                                    test_dataloader=test_dataloader,
                                                                                    lr_scheduler=lr_scheduler)

    logger.info("Engine is built", ranks=[0])

    timer = MultiTimer()

    trainer = Trainer(engine=engine, logger=logger, timer=timer)
    logger.info("Trainer is built", ranks=[0])

    hooks = [
        LogMetricByEpochHook(logger=logger),
        LogMetricByStepHook(),
        # LogTimingByEpochHook(timer=timer, logger=logger),
        # LogMemoryByEpochHook(logger=logger),
        AccuracyHook(accuracy_func=Accuracy()),
        LossHook(),
        ThroughputHook(),
        LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False)
    ]

    logger.info("Train start", ranks=[0])

    begin_time = time.time()


    trainer.fit(train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                epochs=gpc.config.NUM_EPOCHS,
                hooks=hooks,
                display_progress=True,
                test_interval=1)

    end_time = time.time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time)
    print(cuda.max_memory_allocated())


if __name__ == '__main__':
    train_cifar()

