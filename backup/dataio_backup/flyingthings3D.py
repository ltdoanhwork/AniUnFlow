
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any
import random
import cv2
import numpy as np
from .common_readers import read_image, read_flow_any
from .base_dataset import FlowPairDataset
from glob import glob
import os.path as osp
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

        self.occ_list = []
        self.line_list = []

    def __getitem__(self, index):
        # print('Index is {}'.format(index))
        # sys.stdout.flush()
        index = index % len(self.image_list)
        if self.is_test:
            self.augmentor = None

        valid = None
        print(self.flow_list[index])
        flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))            
        else:
            img1 = img1[..., :3]
            
        if len(img2.shape) == 2:
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.is_test:
            occ = torch.from_numpy(np.load(self.occ_list[index]))
            line = torch.from_numpy(np.load(self.line_list[index]))
            return img1, img2, flow, occ, line, self.extra_info[index], valid.float()
        else:
            sample = { "image1": img1, "image2": img2, "flow": flow }
            return sample


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='data/FlyingThings3D_release', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]

# def build_flyingthings3D(aug_params=None, root='data/FlyingThings3D_release', dstype='frames_cleanpass'):
#     return FlyingThings3D(aug_params=aug_params, root=root, dstype=dstype)

def build_flyingthings3D(cfg, split: str):
    aug_params = cfg["aug"]
    root = cfg["dataset"].get("root", "data/FlyingThings3D_release")
    dstype = cfg["dataset"].get("dstype", "frames_cleanpass")
    return FlyingThings3D(aug_params=aug_params, root=root, dstype=dstype)