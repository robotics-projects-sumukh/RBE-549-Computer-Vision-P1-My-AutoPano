import torch

# import torchvision
# from torchvision import transforms, datasets
import torch.nn as nn
# from Network.Network import Net
import cv2
import sys
import numpy as np
import random
import skimage
import PIL
import os
import glob
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
from termcolor import colored, cprint
import math
from tqdm import tqdm
from Network.Supervised_Network import LossFn, SupNet
from Network.Unsupervised_Network import Net
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def GenerateCustomData(imgA, p_size, perturbation_range):

    h, w = imgA.shape[:2]

    x_min = np.random.randint(0, w - p_size)
    y_min = np.random.randint(0, h - p_size)

    x_max = x_min + p_size
    y_max = y_min + p_size

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


def get_EPE(output, homographybatch):
    EPE_loss = torch.nn.MSELoss()
    EPE = EPE_loss(output, homographybatch)
    return EPE


def infer_model(
    model,
    patchbatch,
    pA,
    pB,
    test_img_tensor,
    coordsA,
    type="Supervised",
):
    model.eval()
    if type == "Sup":
        output = model(patchbatch.to(device)) 
        output = output.view(-1, 8) * 32
    else:
        pA = torch.from_numpy(pA)
        pB = torch.from_numpy(pB)

        output = model(pA.unsqueeze(0).unsqueeze(0).to(device), pB.unsqueeze(0).unsqueeze(0).to(device))
        output = output.view(-1, 8)

    return output


def predictCornersSup(
    coordsA, patch, homography, pA, pB, test_img_tensor, model_type, img_num
):
    patch = patch.to(device)
    homography = homography.to(device)

    model = SupNet((2, 128, 128), 8)
    model.load_state_dict(torch.load("../Checkpoints/sup_model.ckpt", map_location=device, weights_only=False)["model_state_dict"])
    model.to(device)
    homography_pred = (
        infer_model(model, patch, pA, pB, test_img_tensor, coordsA, "Sup")
    )
    epe = get_EPE(homography_pred, homography)
    epe = round(epe.item(), 4)
    homography_pred = homography_pred.view(4, 2)
    homography_pred = homography_pred.cpu().detach().numpy()

    coordsB_pred = coordsA + homography_pred

    return coordsB_pred, epe


def predictCornersUnSup(
    coordsA, patch, homography, pA, pB, test_img_tensor, model_type, img_num
):
    patch = patch.to(device)
    homography = homography.to(device)

    model = Net((2, 128, 128), 8)
    model.load_state_dict(torch.load("../Checkpoints/unsup_model_2.ckpt", map_location=device, weights_only=False)["model_state_dict"])
    model.to(device)

    homography_pred = (
        infer_model(model, patch, pA, pB, test_img_tensor, coordsA, "Unsup")
    )
    epe = get_EPE(homography_pred, homography)
    epe = round(epe.item(), 4)
    homography_pred = homography_pred.view(4, 2)
    homography_pred = homography_pred.cpu().detach().numpy()
    coordsB_pred = coordsA + homography_pred

    return coordsB_pred, epe


def saveImage(img, img_num, img_ct, dataset):
    save_img_path = "./OutputImages/" + dataset + "Images/"
    if not os.path.isdir(save_img_path):
        os.makedirs(save_img_path)
    cv2.imwrite(save_img_path + str(img_num) + "_" + str(img_ct) + ".jpg", img)


def visualize(coordsA, coordsB, coordsB_pred_sup, coordsB_pred_un_sup ,coordsB_pred_classical, imageA, img_num, img_ct, dataset):
    coordinates = [coordsA, coordsB, coordsB_pred_sup, coordsB_pred_un_sup, coordsB_pred_classical]
    colors = [(0, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    labels = ['Unwarped Random Crop', 'Warped Random Crop', 'Supervised Warp Estimation', 'Unsupervised Warp Estimation', 'Classical Warp Estimation']
    
    legend_height = 140  
    h, w = imageA.shape[:2]

    new_image = np.zeros((h + legend_height, w, 3), dtype=np.uint8)
    new_image[legend_height:, :] = imageA
    new_image[:legend_height, :] = 255
    
    # Draw coordinate lines
    for e, element in enumerate(coordinates):
        color = colors[e]
        for i in range(len(element)):
            point1 = [int(coord) for coord in element[i]]
            pt1 = (point1[0], point1[1] + legend_height)
            point2 = [int(coord) for coord in element[(i + 1) % len(element)]]
            pt2 = (point2[0], point2[1] + legend_height)
            cv2.line(new_image, pt1, pt2, color, 3)

    # Draw legend vertically
    legend_x = 10
    legend_y = 20
    box_width = 20
    line_height = 22  # Space between lines
    
    for color, label in zip(colors, labels):
        # Draw colored rectangle
        cv2.rectangle(new_image, (legend_x, legend_y), (legend_x + box_width, legend_y + 15), color, -1)
        
        # Add label text
        cv2.putText(
            new_image,
            label,
            (legend_x + box_width + 10, legend_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        
        legend_y += line_height

    saveImage(new_image, img_num, img_ct, dataset)


def predictCorners_classical(
    coordsA, patch, homography, pA, pB, test_img_tensor, model_type, img_num
):

    H, _ = get_panorama(pA, pB)
    if H is None:
        return coordsA, -1

    coords = np.array([[0, 0], [127, 0], [127, 127], [0, 127]])
    homogeneous_coords = np.hstack([coords, np.ones((4, 1))])  # Shape (4, 3)
    transformed = (H @ homogeneous_coords.T).T  # Shape (4, 3)
    transformed_corners = transformed[:, :2] / transformed[:, 2, np.newaxis]
    
    transformed_corners = np.array(transformed_corners)
    transformed_corners = transformed_corners.astype(int)
    delta = transformed_corners - coords
    pred_coordsB = coordsA - delta
    return pred_coordsB, 0

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--DataSet", help="Choose from: Train, Test, Val", default="Test"
    )
    Parser.add_argument(
        "--ModelType",
        help="Choose from: Supervised, Unsupervised",
        default="Supervised",
    )
    Parser.add_argument(
        "--PathToData", help="Provide path to dataset.", default="../Data/Test/Phase2/"
    )
    Parser.add_argument(
        "--ImageSelection",
        help="Choose whether to Randomize Selection of Images to Test: Random or Custom",
        default="Random",
    )
    Parser.add_argument(
        "--ImageNum",
        help="If Image Selection is custom, define an image number to start with.",
        default=1,
    )
    Parser.add_argument(
        "--ImageCount", help="Number of images from image number to test", default=10
    )

    Parser.add_argument(
        "--DeletePreviousResults", help="Delete Previous Results", default=False
    )

    Args = Parser.parse_args()
    DataSet = Args.DataSet
    model_type = Args.ModelType
    DataPath = Args.PathToData
    ImageNum = Args.ImageNum
    ImageCt = int(Args.ImageCount)
    ImageSelection = Args.ImageSelection
    Dpr = Args.DeletePreviousResults

    # Clear Output Folder
    saveOutputPath = "./OutputImages/" + DataSet + "Images/"
    if Dpr:
        if os.path.exists(saveOutputPath):
            shutil.rmtree(saveOutputPath)
        else:
            print("The directory does not exist")

    epe_sup_list = []
    epe_unsup_list = []

    i = 0
    while i < ImageCt:
        j = random.randint(1, len(os.listdir(DataPath)))

        imgA_path = DataPath + str(j) + ".jpg"
        imgA = cv2.imread(imgA_path)

        p_size = 128
        perturbation_range = 32
        patch_A, patch_B, H4pt, corner_coords = GenerateCustomData(
            imgA, p_size, perturbation_range
        )

        # Assign Image Labels
        homography = np.array(H4pt)
        homography = torch.from_numpy(homography)
        homography = homography.view(-1, 8)

        corner_coords = np.array(corner_coords)
        coordsA = [element[0] for element in corner_coords]
        coordsB = [element[1] for element in corner_coords]


        img_tensor = np.array(imgA)
        img_tensor = torch.from_numpy(img_tensor)

        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        imgA_tensor = np.array(imgA)
        imgA_tensor = torch.from_numpy(imgA_tensor)

        imgA_tensor = imgA_tensor.unsqueeze(0)
        imgA_tensor = imgA_tensor.permute(0, 3, 1, 2)

        patch_A_ = patch_A
        patch_B_ = patch_B

        patch_A = cv2.cvtColor(patch_A, cv2.COLOR_BGR2GRAY)
        patch_B = cv2.cvtColor(patch_B, cv2.COLOR_BGR2GRAY)

        pA = np.float32(patch_A)
        pB = np.float32(patch_B)

        pA = pA / 255
        pB = pB / 255

        patch = np.dstack((pA, pB))
        patch = torch.from_numpy(patch)
        patch = patch.permute(2, 0, 1)

        patch = patch.unsqueeze(0)

        coordsB_pred_classical, epe_classical = predictCorners_classical(
            coordsA, patch, homography, patch_A_, patch_B_, imgA_tensor, model_type, j
        )
        if epe_classical == -1:
            continue

        coordsB_pred_sup, epe_sup = predictCornersSup(
            coordsA, patch, homography, pA, pB, imgA_tensor,"Sup", j
        )
        coordsB_pred_un_sup, epe_unsup = predictCornersUnSup(
            coordsA, patch, homography, pA, pB, imgA_tensor,"Unsup", j
        )       

        print("Image Number: ", j, " iteration: ", i + 1) 
        i += 1
                

        epe_sup_list.append(epe_sup)
        epe_unsup_list.append(epe_unsup)

        visualize(coordsA, coordsB, coordsB_pred_sup, coordsB_pred_un_sup, coordsB_pred_classical, imgA, j, i + 1, DataSet)

    print("Average EPE Supervised: ", round(np.mean(epe_sup_list), 4))
    print("Average EPE Unsupervised: ", round(np.mean(epe_unsup_list), 4))


if __name__ == "__main__":
    main()