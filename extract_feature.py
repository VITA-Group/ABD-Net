import torch
from tqdm import tqdm
import argparse
import os
from torchreid import models
from torchreid.utils.torchtools import set_bn_to_eval, count_num_param
from torch.utils.data import DataLoader
from torchreid.eval_625.dataset import evalDataset
from torchreid import transforms as T
import scipy.io


def extractor(model, dataloader):
    def fliplr(img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    model.eval()
    with torch.no_grad():
        test_names = []
        test_features = torch.FloatTensor()

        for batch, sample in enumerate(tqdm(dataloader)):
            names, images = sample['name'], sample['img']

            ff = model(images.cuda()).data.cpu()
            print(ff.shape)
            ff = ff + model(fliplr(images).cuda()).data.cpu()
            ff = ff.div(torch.norm(ff, p=2, dim=1, keepdim=True).expand_as(ff))

            test_names = test_names + names
            test_features = torch.cat((test_features, ff), 0)

    return test_names, test_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap_shot', type=str, default='saved-models/densenet121_xent_market1501.pth.tar')
    parser.add_argument('--arch', type=str, default='densenet121')
    parser.add_argument('--dataset-path', type=str, default='data/valset/valSet')
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image (default: 256)")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image (default: 128)")
    parser.add_argument('--test-batch', default=100, type=int,
                        help="test batch size")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--log-dir', type=str, default='log/eval_625')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    pin_memory = True if args.gpu else False

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=751, loss={'xent'}, use_gpu=args.gpu).cuda()
    print("Model size: {:.3f} M".format(count_num_param(model)))

    checkpoint = torch.load(args.snap_shot)
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Loaded pretrained weights from '{}'".format(args.snap_shot))

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    queryloader = DataLoader(
        evalDataset(os.path.join(args.dataset_path, 'query'), transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        evalDataset(os.path.join(args.dataset_path, 'gallery'), transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    dataloaders = {'query': queryloader,
                   'gallery': galleryloader}

    for dataset in ['val']:
        for subset in ['query', 'gallery']:
            test_names, test_features = extractor(model, dataloaders[subset])
            results = {'names': test_names, 'features': test_features.numpy()}
            scipy.io.savemat(os.path.join(args.log_dir, 'feature_%s_%s.mat' % (dataset, subset)), results)


if __name__ == '__main__':
    main()
