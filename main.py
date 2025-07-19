# Copyright (c) Zhisheng Zheng, The University of Texas at Austin.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Audio-MAE: https://github.com/facebookresearch/AudioMAE
# --------------------------------------------------------
import os, time, datetime, json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import timm
# assert timm.__version__ == "0.3.2"  # version check

import utils.lr_decay as lrd
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from data.dataset import DistributedSamplerWrapper, DistributedWeightedSampler, BinauralEvaDataset

import spatial_eva
from engine_eval import evaluate, train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='build_EVA', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=355, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./outputs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # For audioset
    parser.add_argument("--audio_path_root", type=str, default='/path/to/audioset', help="audioset folder path")
    parser.add_argument("--audioset_train", type=str, default='/path/to/train', help="training data json")
    parser.add_argument("--audioset_eval", type=str, default='/path/to/eval', help="validation data json")
    parser.add_argument("--weight_csv", type=str, default='/path/to/weight', help="weight file")
    
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=192)
    parser.add_argument('--timem', help='time mask max length', type=int, default=48)
    parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["mrsaudio"])
    parser.add_argument("--use_soft", type=bool, default=False)
    parser.add_argument('--audio_normalize', action='store_true', help='normalize the audio')

    #parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument('--first_eval_ep', default=0, type=int, help='do eval after first_eval_ep')
    parser.add_argument('--use_custom_patch', action='store_true', default=False, help='use custom patch with overlapping and override timm PatchEmbed')
    parser.add_argument('--source_custom_patch', action='store_true', default=False, help='the pre-trained model already use custom patch')
    parser.add_argument('--roll_mag_aug', action='store_true', default=False, help='use roll_mag_aug')
    parser.add_argument('--mask_t_prob', default=0.0, type=float, help='T masking ratio (percentage of removed patches).') #  
    parser.add_argument('--mask_f_prob', default=0.0, type=float, help='F masking ratio (percentage of removed patches).') #  
    parser.add_argument('--weight_sampler', action='store_true', default=False, help='use weight_sampler')
    parser.add_argument('--epoch_len', default=200000, type=int, help='num of samples/epoch with weight_sampler')
    parser.add_argument('--distributed_wrapper', action='store_true', default=False, help='use distributedwrapper for weighted sampler')
    parser.add_argument('--replacement', action='store_true', default=False, help='use weight_sampler')
    parser.add_argument('--mask_2d', action='store_true', default=True, help='use 2d masking')
    parser.add_argument('--replace_with_mae', action='store_true', default=False, help='replace_with_mae')
    parser.add_argument('--load_imgnet_pt', action='store_true', default=False, help='when img_pt_ckpt, if load_imgnet_pt, use img_pt_ckpt to initialize audio branch, if not, keep audio branch random')
    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    norm_stats = {'mrsaudio': [-4.2677393, 4.5689974]}
    target_length = {'mrsaudio': 1024 }

    audio_conf_train = {
        'num_mel_bins': 128, 
        'target_length': target_length[args.dataset], 
        'freqm': 48,
        'timem': 192,
        'dataset': args.dataset,
        'mode':'train',
        'mean':norm_stats[args.dataset][0],
        'std':norm_stats[args.dataset][1],
        'noise':False,
    }

    audio_conf_val = {
        'num_mel_bins': 128, 
        'target_length': target_length[args.dataset], 
        'freqm': 0,
        'timem': 0,
        'dataset': args.dataset,
        'mode':'val',
        'mean':norm_stats[args.dataset][0],
        'std':norm_stats[args.dataset][1],
        'noise':False,
    }  

    dataset_train = BinauralEvaDataset(
        audio_conf=audio_conf_train,
        audio_path_root=args.audio_path_root,
        normalize=args.audio_normalize, 
        _ext_audio=".wav",
        mode='train'
    )
    
    dataset_val = BinauralEvaDataset(
        audio_conf=audio_conf_val, 
        audio_path_root=args.audio_path_root,
        normalize=args.audio_normalize, 
        _ext_audio=".wav",
        mode='eval'
    )

    #args.distributed:
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    num_nodes = int(os.environ.get('num_nodes', 1))
    ddp = int(os.environ.get('DDP', 1))
    num_nodes = max(ddp, num_nodes)
    rank = int(os.environ.get('NODE_RANK', 0))
    
    print(f"num_nodes:{num_nodes}, rank:{rank}, ddp:{ddp}, num_tasks:{num_tasks}, global_rank:{global_rank}")
    # num_nodes:1, rank:0, ddp:1, num_tasks:8, global_rank:0 (sbatch)
    if args.weight_sampler:
        samples_weight = np.loadtxt(args.weight_csv, delimiter=',')
        if args.distributed_wrapper:
            print('use distributed_wrapper sampler')
            epoch_len=args.epoch_len #200000 #=> 250000
            #epoch_len=21000 # AS-20K
            # replacement should be False
            sampler_train = DistributedSamplerWrapper(
                                sampler=WeightedRandomSampler(samples_weight, num_samples=epoch_len, replacement=args.replacement),
                                dataset=range(epoch_len),
                                num_replicas=num_tasks, #num_nodes, #num_tasks?
                                rank=global_rank, #rank, # global_rank?
                            )
        else:
            #sampler_train = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
            sampler_train = DistributedWeightedSampler(dataset_train, samples_weight,  num_replicas=num_tasks, rank=global_rank, replacement=args.replacement)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_train.collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=dataset_val.collate_fn,
    )
    
    model = spatial_eva.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        num_cls_tokens=3,
    )

    #if args.finetune and not args.eval:
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        if not args.eval:
            for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"Trainable param: {n}, {p.shape}, {p.dtype}")

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    loss_scaler = NativeScaler()
    criterion = nn.MSELoss()
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.dist_eval, args.output_dir)
        print(f"All embeddings have been extracted!")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_mae = 100.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer,
                args=args
            )
        if args.output_dir and epoch > 35 and (epoch % 1 == 0 or epoch == args.epochs - 1):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if epoch >= args.first_eval_ep:
            test_stats = evaluate(data_loader_val, model, device, args.dist_eval)
            print(f"Overall mae of the network on the {len(dataset_val)} test images: {test_stats['overall_mae']:.4f}")
            min_mae = min(min_mae, test_stats["overall_mae"])
            print(f'Min overall score mae: {min_mae:.4f}')
            print(f"Quality mae: {test_stats['quality_mae']:.4f}")
            print(f"Spatial mae: {test_stats['spatial_mae']:.4f}")
            print(f"Localization mae: {test_stats['localization_mae']:.4f}")
        else:
            test_stats ={'min_mae': 100.0}
            print(f'too new to evaluate!')

        if log_writer is not None:
            log_writer.add_scalar('perf/min_mae', test_stats['overall_mae'], epoch)

        log_stats = {
            'epoch': epoch,
            'n_parameters': n_parameters,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
