import argparse
import cv2
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

PATH_TO_HOVERNET_OUT = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/hovernet_out"
PATH_TO_OUTPUT_CELLS = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/tma_patch_masks"

def main(tma_id):
    # Save cells to HDF5
    print(f"TMA id: {tma_id}")
    cells_hdf5_filename = os.path.join(PATH_TO_OUTPUT_CELLS, f"{tma_id}_cells.hdf5")
    f = h5py.File(cells_hdf5_filename, "w")
    i = 0
    files = [filename for filename in os.listdir(PATH_TO_HOVERNET_OUT)
                       if filename.startswith(tma_id)]
    print(f"Number of files to process: {len(files)}")
    for patch_name in files:
        if i % 1000 == 0:
            print(f"Iteration: {i}")
        path_to_patch = os.path.join(PATH_TO_HOVERNET_OUT, patch_name)
        seg_map = np.load(os.path.join(path_to_patch, "instances.npy"))
        path_to_patch_instance_dict = os.path.join(path_to_patch, "nuclei_dict.json")
        
        with open(path_to_patch_instance_dict) as json_file:
            data = json.load(json_file)
            binary_seg_mask = seg_map.copy()
            binary_seg_mask[binary_seg_mask > 0] = 1

            patch_name_list = patch_name.split("-")
            tma, patient_id, patch_num = patch_name_list[0], patch_name_list[1], patch_name_list[2]
            group_name = f"{tma}/{patient_id}/{patch_num}"
            grp = f.create_group(group_name)
            dset = grp.create_dataset(patch_num, data=binary_seg_mask, dtype='uint8')
        i += 1
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tma_id", default='tma1')
    args = parser.parse_args()
    main(args.tma_id)
    print("Success!")
