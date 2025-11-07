from typing import Optional
import os
import numpy as np
import cv2

# --- Images ---

def read_image(path: str, root: Optional[str] = None):
    path = os.path.join(root, path) if (root and not os.path.isabs(path)) else path
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

# --- Flow ---

def _read_flo(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError('Invalid .flo file')
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        flow = np.reshape(data, (h, w, 2))
    return flow

def _read_pfm(file: str) -> np.ndarray:
    # simple PFM reader (single 2-channel pfm is common in some datasets)
    import re
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').rstrip()
        if header not in ("PF", "Pf"):
            raise Exception('Not a PFM file.')
        dims = f.readline().decode('utf-8')
        while dims.startswith('#'):
            dims = f.readline().decode('utf-8')
        width, height = map(int, re.findall(r'\d+', dims))
        scale = float(f.readline().decode('utf-8').rstrip())
        data = np.fromfile(f, '<f' if scale < 0 else '>f')
        shape = (height, width, 3 if header == 'PF' else 1)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        if shape[2] == 3:
            # If 3-channel PFM provided, use first 2 channels as flow
            data = data[:, :, :2]
        return data.astype(np.float32)

def _read_kitti_png(path: str) -> np.ndarray:
    # KITTI encodes flow in 16-bit PNG with scaling
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    flow = im.astype(np.float32)
    flow = flow / 64.0 - 512.0
    # last dim may be 3 (u, v, valid) – keep u,v
    if flow.ndim == 3:
        flow = flow[..., :2]
    return flow


def read_flow_any(path: str, root: Optional[str] = None) -> np.ndarray:
    path = os.path.join(root, path) if (root and not os.path.isabs(path)) else path
    ext = os.path.splitext(path)[1].lower()
    if ext == '.flo':
        return _read_flo(path)
    if ext == '.pfm':
        return _read_pfm(path)
    if ext == '.npy':
        arr = np.load(path)
        if arr.ndim == 2:
            arr = np.stack([arr[..., 0], arr[..., 1]], axis=-1)
        return arr.astype(np.float32)
    if ext == '.png':
        return _read_kitti_png(path)
    raise ValueError(f"Unsupported flow format: {ext}")