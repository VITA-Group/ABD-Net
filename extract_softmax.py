from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.losses import CrossEntropyLoss, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.optimizers import init_optimizer
from torchreid.regularizers import get_regularizer
from torchreid.losses.wrapped_cross_entropy_loss import WrappedCrossEntropyLoss

from torchreid.models.tricks.dropout import DropoutOptimizer

import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'CRITICAL'))

# global variables
parser = argument_parser()
args = parser.parse_args()
dropout_optimizer = DropoutOptimizer(args)

os.environ['TORCH_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.torch'))


def get_criterions(num_classes: int, use_gpu: bool, args) -> ('criterion', 'fix_criterion', 'switch_criterion'):

    from torchreid.losses.wrapped_triplet_loss import WrappedTripletLoss

    from torchreid.regularizers.param_controller import HtriParamController

    htri_param_controller = HtriParamController()

    if 'htri' in args.criterion:
        fix_criterion = WrappedTripletLoss(num_classes, use_gpu, args, htri_param_controller)
        switch_criterion = WrappedTripletLoss(num_classes, use_gpu, args, htri_param_controller)
    else:
        fix_criterion = WrappedCrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
        switch_criterion = WrappedCrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)

    if args.criterion == 'xent':
        criterion = WrappedCrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    elif args.criterion == 'spectral':
        from torchreid.losses.spectral_loss import SpectralLoss
        criterion = SpectralLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth, penalty_position=args.penalty_position)
    elif args.criterion == 'batch_spectral':
        from torchreid.losses.batch_spectral_loss import BatchSpectralLoss
        criterion = BatchSpectralLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    elif args.criterion == 'lowrank':
        from torchreid.losses.lowrank_loss import LowRankLoss
        criterion = LowRankLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    elif args.criterion == 'singular':
        from torchreid.losses.singular_loss import SingularLoss
        criterion = SingularLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth, penalty_position=args.penalty_position)
    elif args.criterion == 'htri':
        criterion = WrappedTripletLoss(num_classes=num_classes, use_gpu=use_gpu, args=args, param_controller=htri_param_controller)
    elif args.criterion == 'singular_htri':
        from torchreid.losses.singular_triplet_loss import SingularTripletLoss
        criterion = SingularTripletLoss(num_classes, use_gpu, args, htri_param_controller)
    elif args.criterion == 'incidence':
        from torchreid.losses.incidence_loss import IncidenceLoss
        criterion = IncidenceLoss()
    elif args.criterion == 'incidence_xent':
        from torchreid.losses.incidence_xent_loss import IncidenceXentLoss
        criterion = IncidenceXentLoss(num_classes, use_gpu, args.label_smooth)
    else:
        raise RuntimeError('Unknown criterion {!r}'.format(criterion))

    if args.fix_custom_loss:
        fix_criterion = criterion

    if args.switch_loss < 0:
        criterion, switch_criterion = switch_criterion, criterion

    return criterion, fix_criterion, switch_criterion, htri_param_controller


def main():
    global args, dropout_optimizer

    torch.manual_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stderr = sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent'}, use_gpu=use_gpu, dropout_optimizer=dropout_optimizer)
    print(model)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    # criterion = WrappedCrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion, fix_criterion, switch_criterion, htri_param_controller = get_criterions(dm.num_train_pids, use_gpu, args)
    regularizer, reg_param_controller = get_regularizer(args.regularizer)
    optimizer = init_optimizer(model.parameters(), **optimizer_kwargs(args))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)

    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        try:

            checkpoint = torch.load(args.load_weights)
        except Exception as e:
            print(e)
            checkpoint = torch.load(args.load_weights, map_location={'cuda:0': 'cpu'})

        # dropout_optimizer.set_p(checkpoint.get('dropout_p', 0))
        # print(list(checkpoint.keys()), checkpoint['dropout_p'])

        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume and check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        state = model.state_dict()
        state.update(checkpoint['state_dict'])
        model.load_state_dict(state)
        # args.start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint from '{}'".format(args.resume))
        print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch, checkpoint['rank1']))

    if use_gpu:
        model = nn.DataParallel(model, device_ids=list(range(len(args.gpu_devices.split(','))))).cuda()

    extract_train_info(model, trainloader)


def extract_train_info(model, trainloader):

    model.eval()
    os.environ['fake'] = '1'

    score_list = [[], [], []]
    correct_list = [[], [], []]
    ps = []
    for imgs, pids, _, paths in trainloader:

        xent_features = model(imgs)[1]
        for i, xent_feature in enumerate(xent_features):
            print(xent_feature.size())
            scores, indexs = torch.max(F.softmax(xent_feature), 1)
            corrects = indexs == pids.cuda()
            print(corrects)
            score_list[i].extend(scores.data)
            correct_list[i].extend(corrects.data)
            print(paths)
            ps.extend(paths)

    with open('softmax_results.csv', 'w') as f:
        f.write('id filename global_score global_correct p1_score p1_correct p2_score p2_correct\n'.replace(' ', ','))
        for i, xs in enumerate(zip(ps, score_list[0], correct_list[0], score_list[1], correct_list[1], score_list[2], correct_list[2])):
            f.write(','.join(map(str, [i, *[x if isinstance(x, str) else x.item() for x in xs]])) + '\n')


def train(epoch, model, criterion, regularizer, optimizer, trainloader, use_gpu, fixbase=False, switch_loss=False):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    if fixbase or args.fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):

        try:
            limited = float(os.environ.get('limited', None))
        except (ValueError, TypeError):
            limited = 1
        # print('################# limited', limited)

        if not fixbase and (batch_idx + 1) > limited * len(trainloader):
            break

        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        outputs = model(imgs)
        if False and isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)
        # if True or (fixbase and args.fix_custom_loss) or not fixbase and ((switch_loss and args.switch_loss < 0) or (not switch_loss and args.switch_loss > 0)):
        if not fixbase:
            reg = regularizer(model)
            # print('use reg', reg)
            # print('use reg', reg)
            loss += reg
        optimizer.zero_grad()
        loss.backward()

        if args.use_clip_grad and (args.switch_loss < 0 and switch_loss):
            print('Clip!')
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))

        del loss
        del outputs

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

        end = time.time()


if __name__ == '__main__':
    main()
