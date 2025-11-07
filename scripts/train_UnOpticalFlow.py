import os 
import sys
sys.path.append("models/UnOpticalFlow/")

from core.networks import get_model
from core.config import generate_loss_weights_dict
from core.visualize import Visualizer
from core.evaluation import load_gt_flow_kitti, load_gt_mask
from test import test_kitti_2012, test_kitti_2015, test_eigen_depth, test_nyu, load_nyu_test_data

from collections import OrderedDict
import torch
import torch.utils.data
from tqdm import tqdm
import shutil
import pickle
import pdb
import random
import numpy as np
import torch.backends.cudnn as cudnn

from dataio import *


def save_model(iter_, model_dir, filename, model, optimizer):
    torch.save({"iteration": iter_, "model_state_dict": model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(model_dir, filename))

def load_model(model_dir, filename, model, optimizer):
    data = torch.load(os.path.join(model_dir, filename))
    iter_ = data['iteration']
    model.load_state_dict(data['model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    return iter_, model, optimizer

def train(cfg):
    # load model and optimizer
    model = get_model(cfg.mode)(cfg)
    if cfg.multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': cfg.lr}])

    # Load Pretrained Models
    if cfg.resume:
        if cfg.iter_start > 0:
            cfg.iter_start, model, optimizer = load_model(cfg.model_dir, 'iter_{}.pth'.format(cfg.iter_start), model, optimizer)
        else:
            cfg.iter_start, model, optimizer = load_model(cfg.model_dir, 'last.pth', model, optimizer)
    elif cfg.flow_pretrained_model:
        data = torch.load(cfg.flow_pretrained_model)['model_state_dict']
        renamed_dict = OrderedDict()
        for k, v in data.items():
            if cfg.multi_gpu:
                name = 'module.model_flow.' + k
            elif cfg.mode == 'flowposenet':
                name = 'model_flow.' + k
            else:
                name = 'model_pose.model_flow.' + k
            renamed_dict[name] = v
        missing_keys, unexp_keys = model.load_state_dict(renamed_dict, strict=False)
        print(missing_keys)
        print(unexp_keys)
        print('Load Flow Pretrained Model from ' + cfg.flow_pretrained_model)
    if cfg.depth_pretrained_model and not cfg.resume:
        data = torch.load(cfg.depth_pretrained_model)['model_state_dict']
        if cfg.multi_gpu:
            renamed_dict = OrderedDict()
            for k, v in data.items():
                name = 'module.' + k
                renamed_dict[name] = v
            missing_keys, unexp_keys = model.load_state_dict(renamed_dict, strict=False)
        else:
            missing_keys, unexp_keys = model.load_state_dict(data, strict=False)
        print(missing_keys)
        print('##############')
        print(unexp_keys)
        print('Load Depth Pretrained Model from ' + cfg.depth_pretrained_model)
        loss_weights_dict = generate_loss_weights_dict(cfg)
        visualizer = Visualizer(loss_weights_dict, cfg.log_dump_dir)

        # load dataset
        tr_ds = Animerun(root=cfg["data"]["train_root"],
                                    stride_min=cfg["data"]["stride_min"],
                                    stride_max=cfg["data"]["stride_max"],
                                    crop_size=tuple(cfg["data"]["crop_size"]),
                                    color_jitter=tuple(cfg["data"]["color_jitter"]),
                                    do_flip=cfg["data"]["do_flip"],
                                    grayscale_p=cfg["data"]["grayscale_p"],
                                    img_exts=cfg["data"].get("img_exts"),
                                    is_test=False)


        va_ds = Animerun(root=cfg["data"]["val_root"], stride_min=1, stride_max=1,
                                    crop_size=tuple(cfg["data"]["crop_size"]),
                                    color_jitter=None, do_flip=False, grayscale_p=0.0,
                                    img_exts=cfg["data"].get("img_exts"),
                                    is_test=True)
        
        dataloader = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)