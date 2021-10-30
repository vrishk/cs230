import argparse
import cv2
import h5py
import matlab.engine
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.ndimage import binary_fill_holes
from scipy.special import ellipe
from scipy.spatial import ConvexHull

PATH_TO_OUTPUT_CELLS = "/Users/vivekshankar/Documents/aihc-fall21-lymphoma/lymphoma/processed/cells"

def read_image_npy(np_img):
    img = binary_fill_holes(np_img)
    mask = img.copy()
    img = img.astype(np.uint8)
    img[mask] = 255

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    # TODO(vishankar): It's possible that the line below throws errors!
    cnt = contours[0]

    return img, thresh, cnt

# Save the grayscale image
def save_binary_img(img_name, save_dir, thresh):
    img_name = img_name + ".png"
    img_arry = np.zeros(thresh.shape)
    img_arry = (thresh == 255)
    img_arry=img_arry*1
    cv2.imwrite(os.path.join(save_dir, img_name), thresh)

def compute_ellipticalSF(img, thresh, cnt):
    # fit minimum bounded rectangle
    print(cnt)
    rect = cv2.minAreaRect(cnt)
    (rectCoord1, rectCoord2, rotate_angle) = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(rgb_img,[box],0,(0,0,255),1)
    plt.imshow(rgb_img)
    plt.show()

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
    convex_hull = ConvexHull(cnt.squeeze(1))
    hull_perimtr = convex_hull.area
    hull_area = convex_hull.volume

    return hull_area, hull_perimtr

# Calculate min and max feret diameters
def compute_feretDiameter(binaryImg_dir, eng):
    # run Matlab engine
    binaryImg_dir = binaryImg_dir + ".png"
    img = eng.imread(binaryImg_dir);
    os.remove(binaryImg_dir)
    # threshold = eng.graythresh(img);
    # img = eng.im2bw(img, threshold);
    bw = eng.imbinarize(img);
    bw = eng.imfill(bw,'holes')

    eng.workspace['resMin'] = eng.bwferet(bw, 'MinFeretProperties')
    eng.workspace['resMax'] = eng.bwferet(bw, 'MaxFeretProperties')

    resMin = eng.extract_res(eng.workspace['resMin'])
    resMin = eng.table2cell(resMin)
    resMax = eng.extract_res(eng.workspace['resMax'])
    resMax = eng.table2cell(resMax)

    (MinDiameter, MinAngle, MinCoordinates) = resMin
    (MaxDiameter, MaxAngle, MaxCoordinates) = resMax
    return MinDiameter, MaxDiameter, MinAngle, MaxAngle

def compute_derivedSF(shortAxis, longAxis, area, perimt, MinDiameter, MaxDiameter, hull_area, hull_perimtr):
    esf = shortAxis/longAxis
    csf = 4 * np.pi * area / perimt**2
    sf1 = shortAxis / MaxDiameter
    sf2 = MinDiameter / MaxDiameter
    elg = MaxDiameter / MinDiameter
    cvx = np.sqrt(area / hull_area)
    cmpt = 4 * np.pi * area / hull_perimtr
    return esf, csf, sf1, sf2, elg, cvx, cmpt

def main(tma_id, patient_id, patch_id, cell_id, tma_cells_hdf5_file, eng=None):
    print(tma_id)
    tma_id_l, patient_id_l, patch_id_l, cell_id_l, rotate_angle_l = [], [], [], [], []
    short_axis_l, long_axis_l, ellipse_perim_l, ellipse_area_l = [], [], [], []
    rect_center_x_l, rect_center_y_l, rect_width_l, rect_height_l = [], [], [], []
    ellipse_centroid_x_l, ellipse_centroid_y_l, cell_type_l = [], [], []
    hull_area_l, hull_perimeter_l = [], []
    min_diameter_l, max_diameter_l, min_angle_l, max_angle_l = [], [], [], []
    esf_l, csf_l, sf1_l, sf2_l, elg_l, cvx_l, cmpt_l = [], [], [], [], [], [], []

    num_patches_processed = 0
    patient_group = tma_cells_hdf5_file[tma_id]
    patch_group = patient_group[patient_id]
    cells_group = patch_group[patch_id]
    data = cells_group[cell_id][cell_id]
    cell_npy = np.zeros(data.shape)
    data.read_direct(cell_npy)
    try:
        img, thresh, cnt = read_image_npy(cell_npy)
        patch = f"{tma_id}-{patient_id}-{patch_id}-{cell_id}"
        save_dir = os.path.join(PATH_TO_OUTPUT_CELLS, "temp")
        save_binary_img(patch, save_dir, thresh)
        rectCoord1, rectCoord2, rotate_angle, (ellipse_centroid_x, ellipse_centroid_y), \
              short_axis, long_axis, ellipse_perim, ellipse_area = compute_ellipticalSF(img, thresh, cnt)
        (rect_center_x, rect_center_y), (rect_width, rect_height) = rectCoord1, rectCoord2
        hull_area, hull_perimeter = compute_regionalSF(cnt)
        min_diameter, max_diameter, min_angle, max_angle = compute_feretDiameter(os.path.join(save_dir, patch), eng)
        esf, csf, sf1, sf2, elg, cvx, cmpt = compute_derivedSF(
            short_axis, long_axis, ellipse_area, ellipse_perim, min_diameter,
            max_diameter, hull_area, hull_perimeter)
        cell_type = data.attrs['type']
    except Exception as ex:
        print(f"Exception encountered on {tma_id}-{patient_id}-{patch_id}-{cell_id}")
        print(ex)

    tma_id_l.append(tma_id)
    patient_id_l.append(patient_id)
    patch_id_l.append(patch_id)
    cell_id_l.append(cell_id)
    rotate_angle_l.append(rotate_angle)
    short_axis_l.append(short_axis)
    long_axis_l.append(long_axis)
    ellipse_perim_l.append(ellipse_perim)
    ellipse_area_l.append(ellipse_area)
    rect_center_x_l.append(rect_center_x)
    rect_center_y_l.append(rect_center_y)
    rect_width_l.append(rect_width)
    rect_height_l.append(rect_height)
    ellipse_centroid_x_l.append(ellipse_centroid_x)
    ellipse_centroid_y_l.append(ellipse_centroid_y)
    hull_area_l.append(hull_area)
    hull_perimeter_l.append(hull_perimeter)
    min_diameter_l.append(min_diameter)
    max_diameter_l.append(max_diameter)
    min_angle_l.append(min_angle)
    max_angle_l.append(max_angle)
    esf_l.append(esf)
    csf_l.append(csf)
    sf1_l.append(sf1)
    sf2_l.append(sf2)
    elg_l.append(elg)
    cvx_l.append(cvx)
    cmpt_l.append(cmpt)
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
            "rect_center_x": rect_center_x_l,
            "rect_center_y": rect_center_y_l,
            "rect_width": rect_width_l,
            "rect_height": rect_height_l,
            "ellipse_centroid_x": ellipse_centroid_x_l,
            "ellipse_centroid_y": ellipse_centroid_y_l,
            "hull_area": hull_area_l,
            "hull_perimeter": hull_perimeter,
            "min_diameter": min_diameter_l,
            "max_diameter": max_diameter_l,
            "min_angle": min_angle_l,
            "max_angle": max_angle_l,
            "esf": esf_l,
            "csf": csf_l,
            "sf1": sf1_l,
            "sf2": sf2_l,
            "elongation": elg_l,
            "convexity": cvx_l,
            "compactness": cmpt_l,
            "cell_type": cell_type_l}
                    
    cell_shapes_df = pd.DataFrame(data)
    output_filename = "{patch}.csv"
    cell_shapes_df.to_csv(output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tma_id", default='tma1')
    parser.add_argument("--patient_id", default='E0003B_v1')
    parser.add_argument("--patch_id", default='59')
    parser.add_argument("--cell_id", default='4')
    args = parser.parse_args()
    tma_id = args.tma_id
    patient_id = args.patient_id
    patch_id = args.patch_id
    cell_id = args.cell_id

    path_to_tma_cells_hdf5_file = os.path.join(PATH_TO_OUTPUT_CELLS, f"{tma_id}_cells.hdf5")
    print(path_to_tma_cells_hdf5_file)
    tma_cells_hdf5_file = h5py.File(path_to_tma_cells_hdf5_file, "r")

    # start Matlab engine
    # eng = matlab.engine.start_matlab()
    main(tma_id, patient_id, patch_id, cell_id, tma_cells_hdf5_file, None)

    tma_cells_hdf5_file.close()
    print("Success!")
