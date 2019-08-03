#!/usr/bin/env python3

import torch
import argparse
from torchreid import models


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='densenet121_fc512_fd_none_nohead_dan_none_nohead')
    parser.add_argument('--model', default='log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b___sl_0__fcl_False__reg_none__dropout_none__dau_crop__pp_before__size_256__0/checkpoint_ep60.pth.tar')
    parser.add_argument('--dest')
    parser.add_argument('--gpu', type=int, default=1)

    # Mock
    parser.add_argument('--root', type=str, default='data',
                        help="root path to data directory")
    parser.add_argument('-s', '--source-names', type=str, required=True, nargs='+',
                        help="source datasets (delimited by space)")
    parser.add_argument('-t', '--target-names', type=str, required=True, nargs='+',
                        help="target datasets (delimited by space)")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (tips: 4 or 8 times number of gpus)")
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image")
    parser.add_argument('--split-id', type=int, default=0,
                        help="split index (note: 0-based)")
    parser.add_argument('--train-sampler', type=str, default='',
                        help="sampler for trainloader")
    parser.add_argument('--data-augment', type=str, choices=['none', 'crop', 'random-erase', 'color-jitter', 'crop,random-erase', 'crop,color-jitter'], default='crop')
    parser.add_argument('--train-batch-size', default=32, type=int,
                        help="training batch size")
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help="test batch size")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="number of instances per identity")
    parser.add_argument('--cuhk03-labeled', action='store_true',
                        help="use labeled images, if false, use detected images")
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                        help="use classic split by Li et al. CVPR'14")
    parser.add_argument('--use-metric-cuhk03', action='store_true',
                        help="use cuhk03's metric for evaluation")

    return parser.parse_args()


from args import image_dataset_kwargs
from torchreid.data_manager import ImageDataManager
import os
import numpy as np
args = None

from collections import defaultdict


def evaluate(model, loader):

    model.eval()
    with torch.no_grad():

        pids_lst, f_512, f_1024 = [], [], []
        for _, (imgs, pids, _, _) in enumerate(loader):
            imgs = imgs.cuda()
            pids_lst.extend(pids)

            os.environ['NOFC'] = '1'
            features = model(imgs).data.cpu()
            features = features.div(torch.norm(features, p=2, dim=1, keepdim=True).expand_as(features))
            f_1024.append(features)

            os.environ['NOFC'] = ''
            features = model(imgs).data.cpu()
            features = features.div(torch.norm(features, p=2, dim=1, keepdim=True).expand_as(features))
            f_512.append(features)

        f_512 = torch.cat(f_512, 0)
        f_1024 = torch.cat(f_1024, 0)

        dct = defaultdict(lambda: {'512': [], '1024': []})
        print(f_512.size(), f_1024.size())
        for pid, ff512, ff1024 in zip(pids_lst, f_512, f_1024):
            pid = np.asscalar(pid.cpu().numpy())
            print(pid, ff512.size(), ff1024.size())
            dct[pid]['512'].append(ff512)
            dct[pid]['1024'].append(ff1024)

        for pid, mapping in dct.items():
            mapping['512'] = torch.stack(mapping['512'], 0).numpy()
            mapping['1024'] = torch.stack(mapping['1024'], 0).numpy()

            print('dct', pid, mapping['512'].shape, mapping['1024'].shape)
            print(np.linalg.norm(mapping['512'], axis=1), np.linalg.norm(mapping['1024'], axis=1))

    return dct


def main():

    global args
    args = get_args()
    use_gpu = True

    model = models.init_model(name=args.arch, num_classes=751, loss={'xent'}, use_gpu=args.gpu).cuda()

    checkpoint = torch.load(args.model, map_location={'cuda:0': 'cpu'})
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    q = evaluate(model, testloader_dict['market1501']['query'])
    g = evaluate(model, testloader_dict['market1501']['gallery'])

    import os
    import os.path as osp
    import scipy.io

    os.makedirs(args.dest, exist_ok=True)
    os.makedirs(osp.join(args.dest, 'query'), exist_ok=True)

    for pid, mapping in q.items():
        os.makedirs(osp.join(args.dest, 'query', str(pid)))
        scipy.io.savemat(osp.join(args.dest, 'query', str(pid), '512_1024.mat'), mapping)

    os.makedirs(osp.join(args.dest, 'gallery'), exist_ok=True)

    for pid, mapping in g.items():
        os.makedirs(osp.join(args.dest, 'gallery', str(pid)))
        scipy.io.savemat(osp.join(args.dest, 'gallery', str(pid), '512_1024.mat'), mapping)


if __name__ == '__main__':
    main()
