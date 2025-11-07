import os
from glob import glob
import csv
import os.path as osp
from .base_dataset import FlowPairDataset

class Animerun(FlowPairDataset):
    """Animerun dataset reading from a CSV manifest.
    CSV columns: img1,img2,flow (absolute or root-relative paths)
    """
    def __init__ (self, csv_rows, resize: tuple, aug_params: dict, root: str = None, is_test=False,  dstype='Frame_Anime'):
        super(Animerun, self).__init__(csv_rows, resize, aug_params, root)
        self.is_test = is_test
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
                    # self.image_list += [ [image_list[i], image_list[i+1]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                self.occ_list += sorted(glob(osp.join(unmatch_root, scene, '*.npy')))
                self.flow_list += sorted(glob(osp.join(flow_root, scene, 'forward', '*.flo')))
                self.line_list += [ path.replace(unmatch_root, line_root) for path in sorted(glob(osp.join(unmatch_root, scene, '*.npy')))]

        # print('Len of Flow is ', len(self.flow_list))
        # print('Len of Anime is ', len(self.image_list))
        print('Len of Occlusion is ', len(self.occ_list))
        print('Len of Line Area is ', len(self.line_list))

def _read_csv(csv_path):
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"img1": r["img1"], "img2": r["img2"], "flow": r["flow"]})
    return rows

def build_animerun(cfg, split: str):
    csv_path = cfg["dataset"][f"{split}_csv"]
    rows = _read_csv(csv_path)
    print(split)
    ds = Animerun(rows,
                  resize=tuple(cfg["dataset"]["resize"]),
                  aug_params=cfg["dataset"].get("aug", {}),
                  root=cfg["dataset"].get("root"),
                  is_test=(split == "val"))
    return ds