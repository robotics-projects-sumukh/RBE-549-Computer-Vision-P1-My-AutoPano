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
import utils
import kornia

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_video_frames(video_path):
    """
    Reads a video file and returns a list of frames (images).

    Args:
        video_path (str): Path to the video file.

    Returns:
        frames (list of ndarray): List containing each frame as a NumPy array.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print("Error: Cannot open video file:", video_path)
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Total frames read: {len(frames)}")
    return frames

def GenerateCustomData(imgA, imgB):
    # resize imgA and imgB to 128x128
    imgA = cv2.resize(imgA, (128, 128))
    imgB = cv2.resize(imgB, (128, 128))
    return imgA, imgB


def get_EPE(output, homographybatch):
    # Calculate the EPE loss
    EPE_loss = torch.nn.MSELoss()
    # Calculate the EPE loss
    EPE = EPE_loss(output, homographybatch)
    # EPE = torch.sqrt(EPE)
    return EPE


def infer_model(
    model,
    imgA,
    imgB,
    patch,
    type="Sup",
):
    # Set model to evaluation mode
    model.eval()
    # print("aa")
    # print patchbatch type
    # Generate a batch of data
    patch = patch.float()
    if type == "Sup":
        output = model(patch.to(device)) 
        output = output.view(-1, 8) * 32
    else:
        imgA = imgA.float()
        imgB = imgB.float()
        output = model(imgA, imgB)
        output = output.view(-1, 8)

    return output


def predictCorners(
    imgA, imgB, corner_coordsA, model_type, img_num
):
    
    imgA = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)

    imgB = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)

    imgA = imgA / 255
    imgB = imgB / 255     

    patch = np.dstack((imgA, imgB))
    patch = torch.from_numpy(patch)
    patch = patch.permute(2, 0, 1)

    imgA = torch.from_numpy(imgA)
    imgB = torch.from_numpy(imgB)
        # Add patch dimension
    patch = patch.unsqueeze(0)
    if model_type == "Sup":
        model = SupNet((2, 128, 128), 8)
        model.load_state_dict(torch.load("../Checkpoints/sup_model.ckpt", map_location=device, weights_only=False)["model_state_dict"])
    else:
        model = Net((2, 128, 128), 8)
        model.load_state_dict(torch.load("../Checkpoints/unsup_model_1.ckpt", map_location=device, weights_only=False)["model_state_dict"])

    # Set weights_only=False to load the entire model
    model.to(device)

    imgA = imgA.unsqueeze(0).unsqueeze(0).to(device)
    imgB = imgB.unsqueeze(0).unsqueeze(0).to(device)

    homography_pred = (infer_model(model, imgA, imgB, patch, model_type))

    homography_pred = homography_pred.view(4, 2)
    homography_pred = homography_pred.cpu().detach().numpy()

    corner_coordsB = corner_coordsA + homography_pred

    homography_matrix = cv2.getPerspectiveTransform(
        np.float32(corner_coordsA), np.float32(corner_coordsB)
    )

    return homography_matrix, homography_pred


def saveImage(img, img_num, img_ct, dataset):
    save_img_path = "./OutputImages/" + dataset + "Images/"
    if not os.path.isdir(save_img_path):
        os.makedirs(save_img_path)
    cv2.imwrite(save_img_path + str(img_num) + "_" + str(img_ct) + ".jpg", img)
    print("Image saved as: ",save_img_path + str(img_num) + "_" + str(img_ct) + ".jpg")


def get_warped_image(img1, img2, best_H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners_img2 = cv2.perspectiveTransform(corners_img2, best_H)

    corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.vstack((corners_img1, warped_corners_img2))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())

    translate = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)

    result = cv2.warpPerspective(img2, translate.dot(best_H), (xmax - xmin, ymax - ymin))
    result[-ymin:h1 - ymin, -xmin:w1 - xmin] = img1    

    return result

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--DataSet", help="Choose from: Train, Test, Val", default="Train"
    )
    Parser.add_argument(
        "--ModelType",
        help="Choose from: Supervised, Unsupervised",
        default="Supervised",
    )
    Parser.add_argument(
        "--PathToData", help="Provide path to dataset.", default="../Data/Train/"
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
        "--ImageCount", help="Number of images from image number to test", default=1
    )

    # Delete Previous results arg
    Parser.add_argument(
        "--DeletePreviousResults", help="Delete Previous Results", default=False
    )

    Parser.add_argument(
        "--NameOfTestSet", help="unity_hall, tower, trees", default="unity_hall"
    )

    Args = Parser.parse_args()
    DataSet = Args.DataSet
    model_type = Args.ModelType
    DataPath = Args.PathToData
    ImageNum = Args.ImageNum
    ImageCt = int(Args.ImageCount)
    ImageSelection = Args.ImageSelection
    Dpr = Args.DeletePreviousResults
    NameOfTestSet = Args.NameOfTestSet

    # Clear Output Folder
    saveOutputPath = "./OutputImages/" + DataSet + "Images/"
    if Dpr:
        if os.path.exists(saveOutputPath):
            shutil.rmtree(saveOutputPath)
        else:
            print("The directory does not exist")

    if ImageSelection == "Random":
        ImageNum = None

    DataPath = "../Data/Test/Phase2Pano/" + NameOfTestSet + "/"
    imgs = os.listdir(DataPath)
    print("Number of images: ", len(imgs))
    temp_frame = cv2.imread(DataPath + "0.jpg")

    for i in tqdm(range(1, len(imgs))):
        print("Processing Image: ", i)
        imgA = temp_frame
        imgB = cv2.imread(DataPath + str(i) + ".jpg")

        imgA_raw = imgA
        imgB_raw = imgB        

        imgA, imgB = GenerateCustomData(imgA, imgB)
        corner_coordsA = np.array([[0, 0], [127, 0], [127, 127], [0, 127]])

        phase1_homography_matrix, _ = utils.get_panorama(imgA_raw, imgB_raw)

        imgA = np.array(imgA)
        imgB = np.array(imgB) 

        homography_matrix, homography_pred = predictCorners(
            imgA, imgB, corner_coordsA, model_type, 0
        )

        H1, W1 = imgA_raw.shape[:2]
        H2, W2 = imgB_raw.shape[:2]

        h1, w1 = imgA.shape[:2]
        h2, w2 = imgB.shape[:2]

        s_x1, s_y1 = w1 / W1, h1 / H1
        s_x2, s_y2 = w2 / W2, h2 / H2

        S1 = np.array([[s_x1, 0, 0],
                    [0, s_y1, 0],
                    [0, 0, 1]])

        S2_inv = np.array([[1/s_x2, 0, 0],
                        [0, 1/s_y2, 0],
                        [0, 0, 1]])

        H_original = S2_inv @ homography_matrix @ S1

        H_original[0][2] = phase1_homography_matrix[0][2]
        H_original[1][2] = phase1_homography_matrix[1][2]

        stitched_image_orig = get_warped_image(imgB_raw, imgA_raw, H_original)

        temp_frame = stitched_image_orig
        saveImage(temp_frame, i, NameOfTestSet, "Stitched")

if __name__ == "__main__":
    main()