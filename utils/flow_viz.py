# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

# utils/flow_viz.py
import numpy as np

def make_colorwheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col += BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel  # float32 [ncols,3] in 0..255

def _flow_uv_to_colors_safe(u, v, *, convert_to_bgr=False, colorwheel=None):
    if colorwheel is None:
        colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    # Bảo đảm là 2D array
    u = np.asarray(u)
    v = np.asarray(v)
    assert u.ndim == 2 and v.ndim == 2, "u,v must be HxW"

    rad = np.sqrt(u*u + v*v)             # HxW
    a = np.arctan2(-v, -u) / np.pi       # HxW -> [-1,1]
    fk = (a + 1) / 2 * (ncols - 1)       # HxW
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1)
    k1[k1 == ncols] = 0
    f  = fk - k0                         # HxW in [0,1)

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        tmp  = colorwheel[:, i] / 255.0                 # [ncols]
        col0 = tmp[k0]                                   # HxW
        col1 = tmp[k1]                                   # HxW
        col  = (1.0 - f) * col0 + f * col1               # HxW, writeable

        # Thay vì gán in-place bằng mask, dùng np.where an toàn
        inside = (rad <= 1.0)
        col = np.where(inside, 1 - rad * (1 - col), col * 0.75)

        ch = 2 - i if convert_to_bgr else i
        flow_image[..., ch] = np.floor(255.0 * np.clip(col, 0.0, 1.0)).astype(np.uint8)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, robust_percentile=95, convert_to_bgr=False, rad_max=None):
    """
    flow_uv: np.ndarray [H,W,2] (u,v) in pixels
    clip_flow: float or None -> clip đối xứng [-clip_flow, clip_flow]
    robust_percentile: dùng p95 để chuẩn hóa magnitude (ổn định màu)
    rad_max: float or None -> if provided, use this instead of computing from flow
             (useful for consistent coloring across multiple flows)
    """
    flow_uv = np.asarray(flow_uv)
    assert flow_uv.ndim == 3 and flow_uv.shape[2] == 2, "flow must be HxWx2"
    u = flow_uv[..., 0].astype(np.float32).copy()
    v = flow_uv[..., 1].astype(np.float32).copy()

    if clip_flow is not None:
        u = np.clip(u, -clip_flow, clip_flow)
        v = np.clip(v, -clip_flow, clip_flow)

    rad = np.sqrt(u*u + v*v)
    
    if rad_max is None:
        if robust_percentile is not None:
            rad_max = np.percentile(rad, robust_percentile)
        else:
            rad_max = rad.max()
        rad_max = float(rad_max) + 1e-5
    else:
        rad_max = float(rad_max)

    u /= rad_max
    v /= rad_max

    return _flow_uv_to_colors_safe(u, v, convert_to_bgr=convert_to_bgr)


def compute_flow_magnitude_radmax(flows_list, robust_percentile=95):
    """
    Compute shared rad_max from multiple flows for consistent coloring.
    
    Args:
        flows_list: list of np.ndarray, each [H,W,2]
        robust_percentile: percentile to use (default 95 for stability)
    
    Returns:
        float: rad_max value to use for all flows
    """
    all_mags = []
    for flow in flows_list:
        flow = np.asarray(flow)
        if flow.ndim == 3 and flow.shape[2] == 2:
            u = flow[..., 0].astype(np.float32)
            v = flow[..., 1].astype(np.float32)
            mag = np.sqrt(u*u + v*v)
            all_mags.append(mag.flatten())
    
    if all_mags:
        all_mags = np.concatenate(all_mags)
        if robust_percentile is not None:
            rad_max = np.percentile(all_mags, robust_percentile)
        else:
            rad_max = all_mags.max()
        return float(rad_max) + 1e-5
    else:
        return 1e-5
