#!/usr/bin/evn python
import numpy as np
import cv2
import os
from tqdm import tqdm
import time


def saveLabels(homography, corners, labels_path, type="val"):
    np.save(labels_path + "homography_{}_labels.npy".format(type), homography)
    np.save(labels_path + "{}_corner_coordinates.npy".format(type), corners)


def saveImgPatches(i, j, patch_A, patch_B, patches_path):
    save_patch_A = patches_path + "patchA/"
    save_patch_B = patches_path + "patchB/"

    if not os.path.isdir(save_patch_A) or not os.path.isdir(save_patch_B):
        os.makedirs(save_patch_A)
        os.makedirs(save_patch_B)

    cv2.imwrite(save_patch_A + str(i + 1) + "_" + str(j + 1) + ".jpg", patch_A)
    cv2.imwrite(save_patch_B + str(i + 1) + "_" + str(j + 1) + ".jpg", patch_B)


def generate_data(imgA, patch_size, perturbation_range):

    h, w = imgA.shape[:2]

    x_min = np.random.randint(0, w - patch_size)
    y_min = np.random.randint(0, h - patch_size)

    x_max = x_min + patch_size
    y_max = y_min + patch_size

    coords_A = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    )


    perturbation = np.random.randint(
        -perturbation_range, perturbation_range + 1, (4, 2)
    )

    perturbed_coords_B = [
        [x + dx, y + dy] for (x, y), (dx, dy) in zip(coords_A, perturbation)
    ]

    H_inverse = np.linalg.inv(
        cv2.getPerspectiveTransform(
            np.float32(coords_A), np.float32(perturbed_coords_B)
        )
    )

    imgB = cv2.warpPerspective(imgA, H_inverse, (w, h))

    patch_A = imgA[y_min:y_max, x_min:x_max]
    patch_B = imgB[y_min:y_max, x_min:x_max]

    H4pt = (perturbed_coords_B - coords_A).astype(np.float32)

    corner_coords = [(coords_A[i], perturbed_coords_B[i]) for i in range(len(coords_A))]

    return patch_A, patch_B, H4pt, corner_coords


def main():
    homography_val = dict()
    corners = dict()
    patch_size = 128
    perturbation_range = 32
    val_img_path = "../Data/Val/"
    labels_path = "TxtFiles/"
    patches_path = "../Data/GeneratedValData/"
    start_time = time.time()

    for i in tqdm(range(len(os.listdir(val_img_path)))):  #
        imgA = cv2.imread(val_img_path + str(i + 1) + ".jpg")
        imgA = cv2.resize(imgA, (640, 480))
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        print("Computing data for image" + str(i + 1))
        for j in range(2):
            patch_A, patch_B, H4pt, corner_coords = generate_data(
                imgA, patch_size, perturbation_range
            )
            saveImgPatches(i, j, patch_A, patch_B, patches_path)
            homography_val[str(i + 1) + "_" + str(j + 1)] = H4pt
            corners[str(i + 1) + "_" + str(j + 1)] = corner_coords

    homography_val = np.array(homography_val)
    corners = np.array(corners)

    saveLabels(homography_val, corners, labels_path, type="val")

    end_time = time.time()
    print("Total Time: ", end_time-start_time)
    np_test = np.load(
        labels_path + "homography_val_labels.npy", allow_pickle=True
    )

    print(np_test.shape)
    np_test = np_test.item()
    print(len(np_test))
    print(np_test["1_1"])


if __name__=="__main__":
    main()
