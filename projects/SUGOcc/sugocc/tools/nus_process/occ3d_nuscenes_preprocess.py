"""
Code partly taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/data/labels_downscale.py
"""
import numpy as np
from tqdm import tqdm
import numpy.matlib
import os
import glob
import sys
import argparse

import io_data as SemanticKittiIO
from multiprocessing import Pool
class occ3d_nuscenes_cfg:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="semantic kitti config")
        self.parser.add_argument('--nuscenes_root', type=str, default=None)
    
    def parse(self):
        self.cfg = self.parser.parse_args()
        return self.cfg                            

def _downsample_label(label, voxel_size=(240, 144, 240), downscale=4, empty_label=17):
    r"""downsample the labeled data,
    code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
    Shape:
        label, (240, 144, 240)
        label_downscale, if downsample==4, then (60, 36, 60)
    """
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 1 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == empty_label)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count >= empty_t:
            label_downscale[x, y, z] = empty_label if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin != empty_label, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


def majority_pooling(grid, k_size=2):
    result = np.zeros(
        (grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size)
    )
    for xx in range(0, int(np.floor(grid.shape[0] / k_size))):
        for yy in range(0, int(np.floor(grid.shape[1] / k_size))):
            for zz in range(0, int(np.floor(grid.shape[2] / k_size))):

                sub_m = grid[
                    (xx * k_size) : (xx * k_size) + k_size,
                    (yy * k_size) : (yy * k_size) + k_size,
                    (zz * k_size) : (zz * k_size) + k_size,
                ]
                unique, counts = np.unique(sub_m, return_counts=True)
                if True in ((unique != 0) & (unique != 255)):
                    # Remove counts with 0 and 255
                    counts = counts[((unique != 0) & (unique != 255))]
                    unique = unique[((unique != 0) & (unique != 255))]
                else:
                    if True in (unique == 0):
                        counts = counts[(unique != 255)]
                        unique = unique[(unique != 255)]
                value = unique[np.argmax(counts)]
                result[xx, yy, zz] = value
    return result

def process_label(label, mask, voxel_size=(240, 144, 240), downscale=4, label_filename="123"):
    LABEL_ds = _downsample_label(label, voxel_size, downscale)
    mask_cam = _downsample_label(mask, voxel_size, downscale, empty_label=0)
    np.savez(label_filename, semantics=LABEL_ds, mask_camera=mask_cam)
    print("wrote to", label_filename)

def main(cfg):
    p = Pool(48)
    scenes = os.listdir(os.path.join(cfg.nuscenes_root, "gts"))
    for scene in scenes:
        frames = os.listdir(os.path.join(cfg.nuscenes_root, "gts", scene))
        for frame in frames:
            label_path = os.path.join(cfg.nuscenes_root, "gts", 
                                      scene, frame, "labels.npz")
            label = np.load(label_path)
            voxels = label['semantics']
            mask_camera = label['mask_camera']
            downscaling = {"1_2": 2, "1_4": 4, "1_8": 8}
            for scale in downscaling:
                label_filename = label_path.replace('labels.npz', 
                                                    f"labels_{scale}.npz")
                p.apply_async(process_label, (voxels, mask_camera, (200, 200, 16), downscaling[scale], label_filename))

    p.close()
    p.join()
    

if __name__ == "__main__":
    cfg = occ3d_nuscenes_cfg()
    cfg = cfg.parse()
    main(cfg)