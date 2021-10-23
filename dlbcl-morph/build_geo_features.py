import argparse
import cv2
import h5py
import numpy as np
import os
import pandas as pd
from scipy.ndimage import binary_fill_holes
from scipy.special import ellipe

PATH_TO_OUTPUT_CELLS = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed/cells"

def read_image_npy(np_img):
    img = binary_fill_holes(np_img)
    mask = img.copy()
    img = img.astype(np.uint8)
    img[mask] = 255

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    # TODO(vishankar): It's possible that the line below throws errors!
    cnt = contours[0]

    return thresh, cnt

def compute_ellipticalSF(thresh, cnt):
    # fit minimum bounded rectangle
    rect = cv2.minAreaRect(cnt)
    (rectCoord1, rectCoord2, rotate_angle) = rect

    # fit minimum bounded ellipse
    ellipse = cv2.fitEllipseDirect(cnt)  #(x, y), (MA, ma), angle
    ell = cv2.ellipse(thresh,ellipse,(169,169,169),3)
    (ellpCtr_x, ellpCtr_y), (shortAxis, longAxis), angle = ellipse

    # perimeter and area of ellipse
    a = longAxis / 2
    b = shortAxis / 2
    e = np.sqrt(1 - b**2 / a**2)  # eccentricity
    perimt = 4 * a * ellipe(e*e)
    area = np.pi * a * b

    return rectCoord1, rectCoord2, rotate_angle, (ellpCtr_x, ellpCtr_y), shortAxis, longAxis, perimt, area

def compute_regionalSF(cnt):
    # fit convex hull
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return hull_area


def main(tma_id, tma_cells_hdf5_file):
    print(tma_id)
    tma_id_l, patient_id_l, patch_id_l, cell_id_l, rotate_angle_l = [], [], [], [], []
    short_axis_l, long_axis_l, ellipse_perim_l, ellipse_area_l, hull_area_l = [], [], [], [], []
    rect_center_x_l, rect_center_y_l, rect_width_l, rect_height_l = [], [], [], []
    ellipse_centroid_x_l, ellipse_centroid_y_l, cell_type_l = [], [], []
    num_patches_processed = 0
    patient_group = tma_cells_hdf5_file[tma_id]
    patient_ids = patient_group.keys()

    for j,patient_id in enumerate(patient_ids):
        patch_group = patient_group[patient_id]
        patch_ids = patch_group.keys()

        for k, patch_id in enumerate(patch_ids):
            if num_patches_processed % 100 == 0:
                print(f"Number of patches processed: {num_patches_processed}")
            num_patches_processed += 1
            cells_group = patch_group[patch_id]
            cell_ids = cells_group.keys()

            for l,cell_id in enumerate(cell_ids):
                data = cells_group[cell_id][cell_id]
                cell_npy = np.zeros(data.shape)
                data.read_direct(cell_npy)
                try:
                    thresh, cnt = read_image_npy(cell_npy)
                    rectCoord1, rectCoord2, rotate_angle, (ellipse_centroid_x, ellipse_centroid_y), \
                          short_axis, long_axis, ellipse_perim, ellipse_area = compute_ellipticalSF(thresh, cnt)
                    (rect_center_x, rect_center_y), (rect_width, rect_height) = rectCoord1, rectCoord2
                    hull_area = compute_regionalSF(cnt)
                    cell_type = data.attrs['type']
                except Exception as ex:
                    print(f"Exception encountered on {tma_id}-{patient_id}-{patch_id}-{cell_id}")
                    print(ex)
                    continue

                tma_id_l.append(tma_id)
                patient_id_l.append(patient_id)
                patch_id_l.append(patch_id)
                cell_id_l.append(cell_id)
                rotate_angle_l.append(rotate_angle)
                short_axis_l.append(short_axis)
                long_axis_l.append(long_axis)
                ellipse_perim_l.append(ellipse_perim)
                ellipse_area_l.append(ellipse_area)
                hull_area_l.append(hull_area)
                rect_center_x_l.append(rect_center_x)
                rect_center_y_l.append(rect_center_y)
                rect_width_l.append(rect_width)
                rect_height_l.append(rect_height)
                ellipse_centroid_x_l.append(ellipse_centroid_x)
                ellipse_centroid_y_l.append(ellipse_centroid_y)
                cell_type_l.append(cell_type)
    
    data = {"tma_id" : tma_id_l,
            "patient_id": patient_id_l,
            "patch_id": patch_id_l,
            "cell_id": cell_id_l,
            "rotate_angle": rotate_angle_l,
            "short_axis": short_axis_l,
            "long_axis": long_axis_l,
            "ellipse_perim": ellipse_perim_l,
            "ellipse_area": ellipse_area_l,
            "hull_area": hull_area_l,
            "rect_center_x": rect_center_x_l,
            "rect_center_y": rect_center_y_l,
            "rect_width": rect_width_l,
            "rect_height": rect_height_l,
            "ellipse_centroid_x": ellipse_centroid_x_l,
            "ellipse_centroid_y": ellipse_centroid_y_l,
            "cell_type": cell_type_l}
                    
    cell_shapes_df = pd.DataFrame(data)
    output_filename = os.path.join(PATH_TO_OUTPUT_CELLS, f"{tma_id}_cell_shapes.csv")
    cell_shapes_df.to_csv(output_filename)
    print(f"Total number of patches processed: {num_patches_processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tma_id", default='tma1')
    args = parser.parse_args()
    tma_id = args.tma_id
    path_to_tma_cells_hdf5_file = os.path.join(PATH_TO_OUTPUT_CELLS, f"{tma_id}_cells.hdf5")
    print(path_to_tma_cells_hdf5_file)
    tma_cells_hdf5_file = h5py.File(path_to_tma_cells_hdf5_file, "r")
    main(tma_id, tma_cells_hdf5_file)
    tma_cells_hdf5_file.close()
    print("Success!")