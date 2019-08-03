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

from torchreid.components.dropout import DropoutOptimizer

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
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent'}, use_gpu=use_gpu, dropout_optimizer=dropout_optimizer, args=vars(args))
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

    if args.evaluate:
        print("Evaluate only")

        for name in args.target_names:
            print("Evaluating {} ...".format(name))
            queryloader = testloader_dict[name]['query'], testloader_dict[name]['query_flip']
            galleryloader = testloader_dict[name]['gallery'], testloader_dict[name]['gallery_flip']
            distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    start_time = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    train_time = 0
    print("==> Start training")

    if os.environ.get('test_first') is not None:
        for name in args.target_names:
            print("Evaluating {} ...".format(name))
            queryloader = testloader_dict[name]['query'], testloader_dict[name]['query_flip']
            galleryloader = testloader_dict[name]['gallery'], testloader_dict[name]['gallery_flip']
            rank1 = test(model, queryloader, galleryloader, use_gpu)

    if args.fixbase_epoch > 0:
        oldenv = os.environ.get('sa', '')
        os.environ['sa'] = ''
        print("Train {} for {} epochs while keeping other layers frozen".format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            start_train_time = time.time()
            train(epoch, model, fix_criterion, regularizer, optimizer, trainloader, use_gpu, fixbase=True)
            train_time += round(time.time() - start_train_time)

        print("Done. All layers are open to train for {} epochs".format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
        os.environ['sa'] = oldenv

    max_r1 = 0

    for epoch in range(args.start_epoch, args.max_epoch):
        dropout_optimizer.set_epoch(epoch)
        reg_param_controller.set_epoch(epoch)
        htri_param_controller.set_epoch(epoch)
        dropout_optimizer.set_training(True)
        start_train_time = time.time()
        print(epoch, args.switch_loss)
        print(criterion)

        cond = args.switch_loss > 0 and epoch >= args.switch_loss
        cond = cond or (args.switch_loss < 0 and args.switch_loss + args.max_epoch < epoch)
        if cond:
            print('Switch!')
            criterion = switch_criterion
        train(epoch, model, criterion, regularizer, optimizer, trainloader, use_gpu, fixbase=False, switch_loss=cond)
        train_time += round(time.time() - start_train_time)

        if use_gpu:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        save_checkpoint({
            'state_dict': state_dict,
            'rank1': 0,
            'epoch': epoch,
            'dropout_p': dropout_optimizer.p,
        }, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            dropout_optimizer.set_training(False)  # IMPORTANT!

            for name in args.target_names:
                print("Evaluating {} ...".format(name))
                queryloader = testloader_dict[name]['query'], testloader_dict[name]['query_flip']
                galleryloader = testloader_dict[name]['gallery'], testloader_dict[name]['gallery_flip']
                print('!!!!!!!!FC!!!!!!!!')
                os.environ['NOFC'] = ''
                rank1 = test(model, queryloader, galleryloader, use_gpu)
                ranklogger.write(name, epoch + 1, rank1)

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            if max_r1 < rank1:
                print('Save!', max_r1, rank1)
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                    'dropout_p': dropout_optimizer.p,
                }, False, osp.join(args.save_dir, 'checkpoint_best.pth.tar'))

                max_r1 = rank1

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    ranklogger.show_summary()


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
        print(loss)
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


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):

    if os.environ.get('flip_eval'):
        flip_eval = True
    else:
        flip_eval = args.flip_eval

    if flip_eval:
        print('Flip Eval!')

    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, q_paths = [], [], [], []

        if flip_eval:
            enumerator = enumerate(zip(queryloader[0], queryloader[1]))
        else:
            enumerator = enumerate(queryloader[0])

        for batch_idx, package in enumerator:
            end = time.time()

            if flip_eval:
                (imgs0, pids, camids, paths), (imgs1, _, _, _) = package
                if use_gpu:
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                features = (model(imgs0) + model(imgs1)) / 2.0
                # print(features.size())
            else:
                (imgs, pids, camids, paths) = package
                if use_gpu:
                    imgs = imgs.cuda()

                features = model(imgs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            q_paths.extend(paths)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, g_paths = [], [], [], []
        if flip_eval:
            enumerator = enumerate(zip(galleryloader[0], galleryloader[1]))
        else:
            enumerator = enumerate(galleryloader[0])

        for batch_idx, package in enumerator:
            # print('fuck')
            end = time.time()

            if flip_eval:
                (imgs0, pids, camids, paths), (imgs1, _, _, _) = package
                if use_gpu:
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                features = (model(imgs0) + model(imgs1)) / 2.0
                # print(features.size())
            else:
                (imgs, pids, camids, _) = package
                if use_gpu:
                    imgs = imgs.cuda()

                features = model(imgs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            g_paths.extend(paths)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        if os.environ.get('save_feat'):
            import scipy.io as io
            io.savemat(os.environ.get('save_feat'), {'q': qf.data.numpy(), 'g': gf.data.numpy(), 'qt': q_pids, 'gt': g_pids})
            # return

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if os.environ.get('distmat'):
        import scipy.io as io
        io.savemat(os.environ.get('distmat'), {'distmat': distmat, 'qp': q_paths, 'gp': g_paths})

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
