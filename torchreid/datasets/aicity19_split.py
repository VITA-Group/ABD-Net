import os
import glob
import re
import sys
import os.path as osp

from .bases import BaseImageDataset
from torchreid.data_manager import BaseDataManager


class AICity19Split(BaseImageDataset):

    dataset_dir = 'aicity19'

    def __init__(self, root='data', verbose=True, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        self.train = train = self._process_dir(self.train_dir, 'train_with_cam_1.txt', relabel=True)
        self.new_vid_old_cid_val = new_vid_old_cid_val = self._process_dir(self.train_dir, 'val_new_person_old_cam.txt')
        self.new_vid_new_cid_val = new_vid_new_cid_val = self._process_dir(self.train_dir, 'val_new_person_new_cam.txt')
        self.new_vid_old_cid_query = new_vid_old_cid_query = self._process_dir(self.train_dir, 'test_new_person_old_cam.txt')
        self.new_vid_new_cid_query = new_vid_new_cid_query = self._process_dir(self.train_dir, 'test_new_person_new_cam.txt')
        self.train_gallery = train_gallery = self._process_dir(self.train_dir, 'train_id_with_cam.txt')

        if verbose:
            print("=> AICity19 loaded")
            print("==> New VID Old CID")
            self.print_dataset_statistics(train, new_vid_old_cid_query, train_gallery)
            print("==> New VID New CID")
            self.print_dataset_statistics(train, new_vid_new_cid_query, train_gallery)
            print("===> New VID Old CID Val")
            print(self.get_imagedata_info(new_vid_old_cid_val))
            print("===> New VID New CID Val")
            print(self.get_imagedata_info(new_vid_new_cid_val))

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, id_file, relabel=False):

        dataset = []
        ids = set()

        with open(osp.join(self.dataset_dir, id_file), 'r') as f:

            for line in f:
                fn, id, cid = line.strip().split()
                fn = osp.join(dir_path, fn)
                id = int(id)
                cid = int(cid)
                dataset.append([fn, id, cid])
                ids.add(id)

        if relabel:
            dct = {v: k for k, v in enumerate(sorted(ids))}
            for item in dataset:
                dataset[1] = dct[dataset[1]]

        return dataset


class AICity19ImageDataManager(BaseDataManager):
    """
    Image-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root,
                 split_id=0,
                 height=256,
                 width=128,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 data_augment='none',
                 num_instances=4,  # number of instances per identity (for RandomIdentitySampler)
                 cuhk03_labeled=False,  # use cuhk03's labeled or detected images
                 cuhk03_classic_split=False  # use cuhk03's classic split or 767/700 split
                 ):
        super(ImageDataManager, self).__init__()

        from torchreid.dataset_loader import ImageDataset
        from torchreid.datasets import init_imgreid_dataset
        from torchreid.transforms import build_transforms
        from torch.utils.data import DataLoader
        from torchreid.samplers import RandomIdentitySampler

        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.split_id = split_id
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.num_instances = num_instances
        self.cuhk03_labeled = cuhk03_labeled
        self.cuhk03_classic_split = cuhk03_classic_split
        self.pin_memory = True if self.use_gpu else False

        # Build train and test transform functions
        transform_train = build_transforms(self.height, self.width, is_train=True, data_augment=data_augment)
        transform_test = build_transforms(self.height, self.width, is_train=False, data_augment=data_augment)
        # transform_test_flip = build_transforms(self.height, self.width, is_train=False, data_augment=data_augment, flip=True)

        print("=> Initializing TRAIN (source) datasets")
        self.train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split
            )

            for img_path, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                self.train.append((img_path, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        if self.train_sampler == 'RandomIdentitySampler':
            print('!!! Using RandomIdentitySampler !!!')
            self.trainloader = DataLoader(
                ImageDataset(self.train, transform=transform_train),
                sampler=RandomIdentitySampler(self.train, self.train_batch_size, self.num_instances),
                batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=True
            )

        else:
            self.trainloader = DataLoader(
                ImageDataset(self.train, transform=transform_train),
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=True
            )

        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}

        for name in self.target_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split
            )

            self.testloader_dict[name]['new_vid_old_cid_query'] = DataLoader(
                ImageDataset(dataset.new_vid_old_cid_query, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False
            )

            self.testloader_dict[name]['new_vid_old_cid_val'] = DataLoader(
                ImageDataset(dataset.new_vid_old_cid_val, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False
            )

            self.testloader_dict[name]['new_vid_new_cid_query'] = DataLoader(
                ImageDataset(dataset.new_vid_new_cid_query, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False
            )

            self.testloader_dict[name]['new_vid_new_cid_val'] = DataLoader(
                ImageDataset(dataset.new_vid_new_cid_val, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False
            )

            self.testloader_dict[name]['train_gallery'] = DataLoader(
                ImageDataset(dataset.train_gallery, transform=transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=False
            )

            # self.testdataset_dict[name]['query'] = dataset.query
            # self.testdataset_dict[name]['gallery'] = dataset.gallery

        print("\n")
        print("  **************** Summary ****************")
        print("  train names      : {}".format(self.source_names))
        print("  # train datasets : {}".format(len(self.source_names)))
        print("  # train ids      : {}".format(self._num_train_pids))
        print("  # train images   : {}".format(len(self.train)))
        print("  # train cameras  : {}".format(self._num_train_cams))
        print("  test names       : {}".format(self.target_names))
        print("  *****************************************")
        print("\n")
