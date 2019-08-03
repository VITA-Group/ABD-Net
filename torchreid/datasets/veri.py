import os
from glob import glob
import re
import sys
import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset

class VeRi(BaseImageDataset):

    dataset_dir = 'veri'

    def __init__(self, root='data', verbose=True, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        train = self._get_train()
        query, gallery = self._get_query_test()
        if verbose:
            print("=> VeRi loaded")
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

    def _get_train(self):

        files = glob(osp.join(self.train_dir, '*'))

        ids = set()
        fns = defaultdict(list)
        for file in files:
            id = int(re.findall(r'/(\d{4})_', file)[0])
            ids.add(id)
            fns[id].append(file)

        mapping = {v: i for i, v in enumerate(sorted(ids))}

        dataset = []
        for k, fs in fns.items():
            for i, f in enumerate(fs):
                dataset.append((f, mapping[k], i))

        return dataset

    def _get_query_test(self):

        q_files = set(osp.basename(x) for x in glob(osp.join(self.query_dir, '*')))
        t_files = glob(osp.join(self.gallery_dir, '*'))

        q_dataset = []
        t_dataset = []

        id_mapping = defaultdict(int)

        for f in t_files:
            id, cid = map(int, re.findall(r'/(\d{4})_c(\d{3})', f)[0])
            bn = osp.basename(f)
            # cid = id_mapping[id]
            # print(id, cid)

            if bn in q_files:
                q_dataset.append((f, id, cid))
            t_dataset.append((f, id, cid))

            id_mapping[id] += 1

        return q_dataset, t_dataset
