from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave


class ValSet(object):
    dataset_dir = 'valset/valSet'

    def __init__(self, root='data', verbose=True, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_db = osp.join(self.dataset_dir, 'queryInfo.txt')
        self.gallery_db = osp.join(self.dataset_dir, 'galleryInfo.txt')

        # self._check_before_run()

        # train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_db, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_db, relabel=False)
        num_total_pids = num_query_pids
        num_total_imgs = num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> Market1501 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            # print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        # self.train = train
        self.train = None
        self.query = query
        self.gallery = gallery

        self.num_train_pids = 751  # num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, db, relabel=False):

        dirname = re.findall(r'/([^/]+?)Info', db)[0]

        import collections

        camids = collections.defaultdict(lambda: 0)
        dataset = []
        with open(db, 'r') as f:
            for line in f:
                img_hash, pid = line.strip().split()
                pid = int(pid)
                camid = camids[pid]
                camids[pid] += 1
                dataset.append(
                    (
                        osp.join(self.dataset_dir, dirname, img_hash + '.png'),
                        pid,
                        camid
                    )
                )
                print(osp.join(self.dataset_dir, dirname, img_hash + '.png'))
        print(db, self.dataset_dir)

        return dataset, len(camids), len(dataset)

        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')

        # pid_container = set()
        # for img_path in img_paths:
        #     pid, _ = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue  # junk images are just ignored
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # dataset = []
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue  # junk images are just ignored
        #     assert 0 <= pid <= 1501  # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1  # index starts from 0
        #     if relabel:
        #         pid = pid2label[pid]
        #     dataset.append((img_path, pid, camid))

        # num_pids = len(pid_container)
        # num_imgs = len(dataset)
        # return dataset, num_pids, num_imgs
