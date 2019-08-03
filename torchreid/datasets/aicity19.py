import os
import glob
import re
import sys
import os.path as osp

from .bases import BaseImageDataset

class AICity19(BaseImageDataset):

    dataset_dir = 'aicity19'

    def __init__(self, root='data', verbose=True, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        train = self.get_train()  # _process_dir(self.train_dir, 'train_track.txt')
        gallery = self._process_dir(self.gallery_dir, 'test_track.txt')

        # FAKE! Since official query set has no labels.
        query = self.get_query(gallery)

        if verbose:
            print("=> AICity19 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

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

    def get_train(self):

        from lxml import etree
        with open(osp.join(self.dataset_dir, 'train_label.xml'), 'rb') as f:
            root = etree.parse(f).getroot()

        ids = set()

        dataset = []
        for item in root.iter('Item'):
            fn, id, cid = map(item.get, 'imageName vehicleID cameraID'.split())
            id = int(id)
            cid = int(cid[1:])
            dataset.append([osp.join(self.train_dir, fn), id, cid])
            ids.add(id)

        mapping = {v: k for k, v in enumerate(sorted(ids))}
        for item in dataset:
            item[1] = mapping[item[1]]

        return dataset

    def get_query(self, gallery):

        dct = {}
        for item in gallery:
            dct[item[1]] = item

        return list(dct.values())

    def _process_dir(self, dir_path, id_file, relabel=False):

        with open(osp.join(self.dataset_dir, id_file), 'r') as f:
            dct = dict(enumerate(map(str.split, f.readlines())))

        dataset = []
        for index_, fn_list in dct.items():
            for cid, fn in enumerate(fn_list):
                dataset.append((osp.join(dir_path, fn), index_, cid))

        return dataset
