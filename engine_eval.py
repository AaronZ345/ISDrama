# Copyright (c) Zhisheng Zheng, The University of Texas at Austin.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Audio-MAE: https://github.com/facebookresearch/AudioMAE
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.stat import calculate_stats, concat_all_gather
import os

def train_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        log_writer=None, args=None
    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        waveforms = batch[0]
        metrics_dict = batch[1]

        quality_score = metrics_dict['quality_score'].float().to(device, non_blocking=True)
        spatial_score = metrics_dict['spatial_score'].float().to(device, non_blocking=True)
        localization_score = metrics_dict['localization_score'].float().to(device, non_blocking=True)
        overall_score = metrics_dict['overall_score'].float().to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        outputs = model(waveforms, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob)
        loss1 = criterion(outputs[0], quality_score)
        loss2 = criterion(outputs[1], spatial_score)
        loss3 = criterion(outputs[2], localization_score)
        loss4 = criterion(outputs[3], overall_score)
        loss = loss1 + loss2 + loss3 + loss4
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, dist_eval=False, output_dir=''):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    quality_preds, spatial_preds, localization_preds, overall_preds = [], [], [], []
    quality_targets, spatial_targets, localization_targets, overall_targets = [], [], [], []

    for batch in metric_logger.log_every(data_loader, 300, header):        
        waveforms = batch[0].to(device)
        audio_paths = batch[1]
        quality_pred, spatial_pred, localization_pred, overall_pred, distance_pred, azimuth_pred, elevation_pred = model(waveforms)
        
        for i in range(len(audio_paths)):
            if 'testset' in audio_paths[i]:
                item_name = audio_paths[i].split('/')[-2]
            else:
                item_name = audio_paths[i].split('/')[-1][:-4]

            dis_npy_path = os.path.join(output_dir, item_name+'_dis.npy')
            azi_npy_path = os.path.join(output_dir, item_name+'_azi.npy')
            np.save(dis_npy_path, distance_pred[i].unsqueeze(0).T.cpu().numpy())
            np.save(azi_npy_path, azimuth_pred[i].unsqueeze(0).T.cpu().numpy())
            
    return {
        "quality_mae": 0.0,
        "spatial_mae": 0.0,
        "localization_mae": 0.0,
        "overall_mae": 0.0
    }
