import cv2
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

PATH_TO_HOVERNET_OUT = "/sailhome/vivek96/lymphoma/processed/hovernet_out"
PATH_TO_SAMPLE_PATCH = os.path.join(PATH_TO_HOVERNET_OUT, "tma5-E0683B_v1-39")
PATH_TO_SAMPLE_PATCH_SEG_MAP = os.path.join(PATH_TO_SAMPLE_PATCH, "instances.npy")
PATH_TO_SAMPLE_PATCH_IMAGE = os.path.join(PATH_TO_SAMPLE_PATCH, "overlay.png")
PATH_TO_SAMPLE_PATCH_INSTANCE_DICT = os.path.join(PATH_TO_SAMPLE_PATCH, "nuclei_dict.json")

PATH_TO_OUTPUT_CELLS = "/sailhome/vivek96/lymphoma/processed/cells"

def crop_im(im):
    arg_ones = np.argwhere(im) # shape: (pixels_of_ones, 2)
    row_min, col_min = np.amin(arg_ones, axis=0)
    row_max, col_max = np.amax(arg_ones, axis=0)
    
    return im[row_min:row_max+1, col_min:col_max+1]

def main():
    # Save cells to HDF5
    cells_hdf5_filename = os.path.join(PATH_TO_OUTPUT_CELLS, "cell.hdf5")
    f = h5py.File(cells_hdf5_filename, "w")
    i = 0
    for patch_name in os.listdir(PATH_TO_HOVERNET_OUT):
        if i % 1000 == 0:
            print(f"Iteration: {i}")
        path_to_patch = os.path.join(PATH_TO_HOVERNET_OUT, patch_name)
        seg_map = np.load(os.path.join(path_to_patch, "instances.npy"))
        path_to_patch_instance_dict = os.path.join(path_to_patch, "nuclei_dict.json")
        with open(path_to_patch_instance_dict) as json_file:
            data = json.load(json_file)
            for cell_num in data.keys():
                cell_num_int = int(cell_num)
                cell_mask = (seg_map == cell_num_int)
                im_cell = crop_im(cell_mask)
                im_cell = im_cell.astype(np.bool_)
                patch_name_list = patch_name.split("-")
                tma, patient_id, patch_num = patch_name_list[0], patch_name_list[1], patch_name_list[2]
                group_name = f"{tma}/{patient_id}/{patch_num}/{cell_num}"
                grp = f.create_group(group_name)
                dset = grp.create_dataset(cell_num, data=im_cell, dtype='uint8')
                dset.attrs['centroid'] = data[cell_num]['centroid']
                dset.attrs['contour'] = data[cell_num]['contour']
                dset.attrs['type'] = data[cell_num]['type']
                dset.attrs['probs'] = data[cell_num]['probs']
        i += 1
    f.close()

if __name__ == "__main__":
    main()
