from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import os
from glob import glob
import csv
import os.path as osp
from .datasets_unsup import UnlabeledPairDataset

class Animerun(UnlabeledPairDataset):
    """Animerun dataset reading from a CSV manifest.
    CSV columns: img1,img2,flow (absolute or root-relative paths)
    """
    def __init__ (self, root: str, stride_min: int = 1, stride_max: int = 2,
                        crop_size: Tuple[int,int] = (368,768), 
                        color_jitter: Optional[Tuple[float,float,float,float]] = None,
                        do_flip: bool = True, 
                        grayscale_p: float = 0.0, 
                        img_exts: List[str] | None = None, 
                        is_test=False, 
                        dstype='Frame_Anime'):
        super(Animerun, self).__init__(root=root,
                                       stride_min=stride_min,
                                       stride_max=stride_max,
                                       crop_size=crop_size,
                                       color_jitter=color_jitter,
                                       do_flip=do_flip,
                                       grayscale_p=grayscale_p,
                                       is_test=is_test)
        if is_test:
            split = "test"
        else:
            split = "train"
        self.extra_info = []
        self.occ_list = []
        self.line_list = []
        self.flow_list = []
        self.image_list = []
        flow_root = osp.join(root, split, 'Flow')
        image_root = osp.join(root, split, dstype)
        unmatch_root = osp.join(root, split, 'UnmatchedForward')
        line_root = osp.join(root, split, 'LineArea')

        for scene in os.listdir(image_root):
            for color_pass in os.listdir(osp.join(image_root, scene)):
                if color_pass != 'original':
                    continue
                image_list = sorted(glob(osp.join(image_root, scene, color_pass, '*.png')))
                for i in range(len(image_list)-1):
                    self.image_list += [ [image_list[i], image_list[i+1]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                self.occ_list += sorted(glob(osp.join(unmatch_root, scene, '*.npy')))
                self.flow_list += sorted(glob(osp.join(flow_root, scene, 'forward', '*.flo')))
                self.line_list += [ path.replace(unmatch_root, line_root) for path in sorted(glob(osp.join(unmatch_root, scene, '*.npy')))]

        print('Len of Flow is ', len(self.flow_list))
        print('Len of Anime is ', len(self.image_list))
        print('Len of Occlusion is ', len(self.occ_list))
        print('Len of Line Area is ', len(self.line_list))

