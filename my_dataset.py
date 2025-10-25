from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import glob
import random
import json
import h5py
import numpy as np

class MyDataSet(Dataset):
    """Definition of dataset for biomarker prediction"""

    def __init__(self, result_txt_path, label_path, teacher_emds_path_1, teacher_emds_path_2, teacher_emds_path_3, mode='train', transform=None):
        
        self.samples = []
        self.labels = {}
        self.wsi_emds_1 = {}
        self.wsi_emds_2 = {}
        self.wsi_emds_3 = {}
        self.result_txt_path = result_txt_path
        self.label_path = label_path
        self.teacher_emds_path_1 = teacher_emds_path_1
        self.teacher_emds_path_2 = teacher_emds_path_2
        self.teacher_emds_path_3 = teacher_emds_path_3
        self.mode = mode
        self.transform = transform
        self._load_labels()
        self._load_teacher_emds_1()
        self._load_teacher_emds_2()
        self._load_teacher_emds_3()
        self._pre_process()


    def _load_labels(self):
        with open(self.label_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            key, value = line.strip().split(',')
            self.labels[key] = value


    def _load_teacher_emds_1(self):
        wsi_emds_file_list = os.listdir(self.teacher_emds_path_1)
        for wsi_emds_file in wsi_emds_file_list:
            wsi_emds_file_path = os.path.join(self.teacher_emds_path_1, wsi_emds_file)
            f_wsi_emds = h5py.File(wsi_emds_file_path, "r")
            wsi_emds = np.array(f_wsi_emds['features'])
            wsi_name, _ = os.path.splitext(wsi_emds_file)
            self.wsi_emds_1[wsi_name] = wsi_emds


    def _load_teacher_emds_2(self):
        wsi_emds_file_list = os.listdir(self.teacher_emds_path_2)
        for wsi_emds_file in wsi_emds_file_list:
            wsi_emds_file_path = os.path.join(self.teacher_emds_path_2, wsi_emds_file)
            f_wsi_emds = h5py.File(wsi_emds_file_path, "r")
            wsi_emds = np.array(f_wsi_emds['features'])
            wsi_name, _ = os.path.splitext(wsi_emds_file)
            self.wsi_emds_2[wsi_name] = wsi_emds


    def _load_teacher_emds_3(self):
        wsi_emds_file_list = os.listdir(self.teacher_emds_path_3)
        for wsi_emds_file in wsi_emds_file_list:
            wsi_emds_file_path = os.path.join(self.teacher_emds_path_3, wsi_emds_file)
            f_wsi_emds = h5py.File(wsi_emds_file_path, "r")
            wsi_emds = np.array(f_wsi_emds['features'])
            wsi_name, _ = os.path.splitext(wsi_emds_file)
            self.wsi_emds_3[wsi_name] = wsi_emds


    def _pre_process(self):
        self.raw_samples = os.listdir(self.result_txt_path)
        for raw_sample in self.raw_samples:
            raw_sample_key = raw_sample[:12]
            raw_sample_wsi_name, _, _ = raw_sample.partition("kmeans")
            label = int(self.labels[raw_sample_key])
            soft_emds_1 = self.wsi_emds_1[raw_sample_wsi_name]
            soft_emds_2 = self.wsi_emds_2[raw_sample_wsi_name]
            soft_emds_3 = self.wsi_emds_3[raw_sample_wsi_name]
            raw_sample_path = os.path.join(self.result_txt_path, raw_sample)
            self.samples.append((raw_sample_path, label, soft_emds_1, soft_emds_2, soft_emds_3))
        if self.mode == 'train':
            random.shuffle(self.samples)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        with open(self.samples[index][0], 'r') as f:
             wsi_feat = json.load(f)
        if self.transform is not None:
            wsi_feat = self.transform(wsi_feat)
        wsi_feat = torch.Tensor(wsi_feat)
        return wsi_feat, self.samples[index][1], self.samples[index][2], self.samples[index][3], self.samples[index][4]

    @staticmethod
    def collate_fn(batch):
        # Reference: official implementation in pytorch
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        feats, labels, soft_emds_1, soft_emds_2, soft_emds_3 = tuple(zip(*batch))

        feats = torch.stack(feats, dim=0)
        # print(labels.shape)
        labels = torch.as_tensor(labels)
        soft_emds_1 = torch.as_tensor(np.stack(soft_emds_1))
        soft_emds_2 = torch.as_tensor(np.stack(soft_emds_2))
        soft_emds_3 = torch.as_tensor(np.stack(soft_emds_3))
        
        return feats, labels, soft_emds_1, soft_emds_2, soft_emds_3
