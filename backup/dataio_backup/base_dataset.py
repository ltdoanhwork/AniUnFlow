import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any
import random
import cv2
import numpy as np
from .common_readers import read_image, read_flow_any
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

class FlowPairDataset(Dataset):
    """Generic <img1, img2, flow12> dataset reading from a CSV manifest.
    CSV columns: img1,img2,flow (absolute or root-relative paths)
    """
    def __init__(self, csv_rows, resize: Tuple[int, int], aug_params: Dict[str, Any], root: str = None, sparse=False, is_test=False):
        self.rows = csv_rows
        self.root = root
        self.H, self.W = resize
        self.aug = aug_params or {}
        self.aug_params = {'crop_size': self.aug.get("crop_size", (256, 256)), 'min_scale': self.aug.get("min_scale", -0.2), 'max_scale': self.aug.get("max_scale", 0.6), 'do_flip': self.aug.get("do_flip", True)}
        if sparse:
            self.augmentor = SparseFlowAugmentor(**self.aug_params)
        else:
            self.augmentor = FlowAugmentor(**self.aug_params)
        self.is_test = is_test

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p1, p2, pf = r["img1"], r["img2"], r["flow"]
        im1 = read_image(p1, root=self.root) # HWC, uint8
        im2 = read_image(p2, root=self.root)
        flow = read_flow_any(pf, root=self.root) # HWC, float32, last dim=2
        valid = None
        if self.is_test:
            self.augmentor = None

        # resize to (H, W) with area (images) & scale flow accordingly
        # im1 = cv2.resize(im1, (self.W, self.H), interpolation=cv2.INTER_AREA)
        # im2 = cv2.resize(im2, (self.W, self.H), interpolation=cv2.INTER_AREA)
        scale_x = self.W / flow.shape[1]
        scale_y = self.H / flow.shape[0]
        # flow = cv2.resize(flow, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= scale_x
        flow[..., 1] *= scale_y
        if self.augmentor is not None:
            im1, im2, flow = self.augmentor(im1, im2, flow)
        # to tensor
        tens1 = torch.from_numpy(im1.transpose(2, 0, 1)).float() / 255.0
        tens2 = torch.from_numpy(im2.transpose(2, 0, 1)).float() / 255.0
        flow = torch.from_numpy(flow.transpose(2, 0, 1)).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = ((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).to(torch.bool)
        sample = { "image1": tens1, "image2": tens2, "flow": flow }
        if self.is_test:
            # occ  = cv2.resize(occ, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            occ = torch.from_numpy(np.load(self.occ_list[idx]))
            line = torch.from_numpy(np.load(self.line_list[idx]))
            # flat = cv2.resize(flat, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

            return {"image1": tens1, "image2": tens2, "flow": flow, "occ": occ, "line": line, "extra_info": self.extra_info[idx], "valid": valid.float()}
        return sample