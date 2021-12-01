import argparse
import os
import sys
import random
import json
from os.path import join
from typing import Iterable, Optional

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import (
    DataLoader, BatchSampler, RandomSampler,
    SequentialSampler, DistributedSampler
)

import util
from configs import dynamic_load
from datasets import build_dataset
from models import build_model
from losses import build_criterion
from common import Logger, MetricLogger, SmoothedValue, NoGradientError


DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
    epoch, loader, model, criterion, optimizer, 
    max_norm=0., print_freq=10, tb_logger=None
):
    model.train()
    criterion.train()

    logger = MetricLogger(delimiter='  ')
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for data in logger.log_every(loader, print_freq, header):
        bitmap0, bitmap1 = data['bitmap0'].to(DEV), data['bitmap1'].to(DEV) 
        kpts0, kpts1 = data['keypoints0'].to(DEV), data['keypoints1'].to(DEV)
        desc0, desc1 = data['descriptors0'].to(DEV), data['descriptors1'].to(DEV)
        scores0, scores1 = data['scores0'].to(DEV), data['scores1'].to(DEV)
        ori_im0_shapes, ori_im1_shapes = data['ori_im0_shapes'].to(DEV), data['ori_im1_shapes'].to(DEV)
        assignment = data['assignment'].to(DEV)

        data_dev = {
        'ori_im0_shapes': ori_im0_shapes,
        'ori_im1_shapes': ori_im1_shapes,
        'keypoints0': kpts0,
        'keypoints1': kpts1,
        'descriptors0': desc0,
        'descriptors1': desc1,
        'scores0': scores0,
        'scores1': scores1
    }
        pred = model(data_dev)
        
        try:
            loss_dict = criterion(pred, assignment)
            loss = loss_dict['loss']
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm
                )

            optimizer.step()

        except NoGradientError:
            print('Got error in training, please check codes.')
            sys.exit(1)
        
        loss_dict_reduced = util.reduce_dict(loss_dict)
        loss_dict_reduced_item = {
            k: v.item() for k, v in loss_dict_reduced.items()
        }
        # Update logger and tensorboard_logger
        logger.update(**loss_dict_reduced_item)
        logger.update(lr=optimizer.param_groups[0]["lr"])
        if tb_logger is not None:
            if util.is_main_process():
                tb_logger.add_scalars(loss_dict_reduced, prefix='train')

    logger.synchronize_between_processes()
    print('Averaged stats: ', logger)
    return {k: meter.global_avg for k, meter in logger.meters.items()}


def main(args):
    util.init_distributed_mode(args)

    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    random.seed(seed)
    print('Seed used: ', seed)

    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable params:', n_params)
    model = model.to(DEV)

    critertion, metrics = build_criterion(args)
    critertion = critertion.to(DEV)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    optimizer = torch.optim.Adam(
        model_without_ddp.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=1e-6
    )
    
    train_dataset, test_dataset = build_dataset(args)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)

    batch_train_sampler = BatchSampler(
        train_sampler, args.batch_size, drop_last=True
    )
    dataloader_kwargs = {
        'collate_fn': train_dataset.collate_fn,
        'pin_memory': True,
        'num_workers': 4,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_train_sampler,
        **dataloader_kwargs
    )

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     sampler=test_sampler,
    #     drop_last=True,
    #     **dataloader_kwargs
    # ) 

    if args.load is not None:
        state_dict = torch.load(args.load)
        model.load_state_dict(state_dict['model'])

    # NOTE: Change file name
    artifact_name = (
        f'net_{args.backbone}_module_{args.module_name}_' +
        f'featdim_{args.feature_dim}_numlayers_{args.num_layers}_' +
        f'scoretype_{args.score_type}_losstype_{args.loss}_' +
        f'limit_{args.train_limit}'
    )

    artifact_path = join(args.artifact, artifact_name)
    os.makedirs(artifact_path, exist_ok=True)

    # Build tensorboard
    tb_logger = Logger(artifact_path) if util.is_main_process() else None

    print('Start training...')
    for epoch in range(args.n_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train(
            epoch, train_loader, model, critertion, optimizer,
            max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            tb_logger=tb_logger
        )
        scheduler.step()

        if epoch % args.save_interval == 0 or epoch == args.n_epochs - 1:
            if util.is_main_process():
                torch.save({
                    'model':model_without_ddp.state_dict()
                }, f'{artifact_path}/model-epoch{epoch}.pth')
        
        log_stats = {
            'epoch':epoch,
            'n_params':n_params,
            **{f'train_{k}': v for k, v in train_stats.items()}
        }
        with open(f'{artifact_path}/train.log', 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
    
    print('Train Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_name', type=str, help='Path to config',
        default='matchnet_config'
    )

    global_cfgs = parser.parse_args()
    args = dynamic_load(global_cfgs.config_name)
    prm_str = 'Arguments:\n' + '\n'.join(
        ['{} {}'.format(k.upper(), v) for k, v in vars(args).items()]
    )

    print(prm_str + '\n', '=='*40 + '\n')
    main(args)