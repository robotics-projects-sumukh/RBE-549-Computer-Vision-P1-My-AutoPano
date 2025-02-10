#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Network.Supervised_Network import LossFn, SupNet
from Network.Unsupervised_Network import Net
 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)  

def GenerateBatch(generate_batch_info, MiniBatchSize, ModelType):

    if ModelType == 'Sup':
            
        train_batch = []
        patch_a = generate_batch_info["patch_a"]
        patch_b = generate_batch_info["patch_b"]
        patch_a_list = generate_batch_info["patch_a_list"]
        patch_b_list = generate_batch_info["patch_b_list"]

        ImageNum = 0
        patchAbatch = []
        patchBbatch = []
        patchbatch = []
        homographybatch = []
        while ImageNum < MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(patch_a_list) - 1)
            PatchAName = patch_a + patch_a_list[RandIdx]
            PatchBName = patch_b + patch_b_list[RandIdx]

            img_key_id = PatchAName.split("/")[-1].split(".")[0]
            ImageNum += 1

            imgA = cv2.imread(PatchAName)
            imgB = cv2.imread(PatchBName)

            imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
            imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

            imgA = np.float32(imgA)
            imgB = np.float32(imgB)

            imgA = imgA / 255
            imgB = imgB / 255

            img = np.dstack((imgA, imgB))
            img = torch.from_numpy(img)

            homography = generate_batch_info["homography"][str(img_key_id)]
            homography = torch.from_numpy(homography)
            # Convert 4*2 to 8*1
            homography = homography.view(-1, 8)
            # Reshape from 1*8 to 8
            homography = homography.squeeze(0)
            # Normalize
            homography = homography / 32
            img = img.permute(2, 0, 1)
            patchbatch.append(img)
            homographybatch.append(homography)
        homographybatch = torch.stack(homographybatch)
        patchbatch = torch.stack(patchbatch)
        train_batch.append(homographybatch)
        return patchbatch, homographybatch
    else:
        train_batch = []
        # {"patch_a": train_patch_a_path, "patch_b": train_patch_b_path, "patch_a_list": train_patch_a_list, "patch_b_list": train_patch_b_list}
        patch_a = generate_batch_info["patch_a"]
        patch_b = generate_batch_info["patch_b"]
        patch_a_list = generate_batch_info["patch_a_list"]
        patch_b_list = generate_batch_info["patch_b_list"]
        base_image_path = generate_batch_info["base_image_path"]
        base_image_path_list = generate_batch_info["base_image_path_list"]

        ImageNum = 0
        patchAbatch = []
        patchBbatch = []
        patchbatch = []
        imgbatch = []
        homographybatch = []
        cornerbatch = []
        while ImageNum < MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(patch_a_list) - 1)
            PatchAName = patch_a + patch_a_list[RandIdx]
            PatchBName = patch_b + patch_b_list[RandIdx]
            img_key_id = PatchAName.split("/")[-1].split(".")[0]
            base_img_key = img_key_id.split("_")[0]
            ImageNum += 1

            img1 = cv2.imread(base_image_path + base_img_key + ".jpg")
            imgA = cv2.imread(PatchAName)
            imgB = cv2.imread(PatchBName)

            # Form a 2 channel image
            imgA = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
            imgB = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1 = cv2.resize(img1, (640, 480))
            img1 = img1 / 255

            # Convert to float32
            imgA = np.float32(imgA)
            imgB = np.float32(imgB)

            imgA = imgA / 255
            imgB = imgB / 255


            img = np.dstack((imgA, imgB))
            img1 = torch.from_numpy(img1)
            img1 = img1.unsqueeze(0)
            img = torch.from_numpy(img)

            homography = generate_batch_info["homography"][str(img_key_id)]
            corner_coordinates = generate_batch_info["corner_coordinates"][str(img_key_id)]
            temp_corners = []
            for corner in corner_coordinates:
                temp_corner = corner[0]
                # Convert to float
                temp_corner = [float(i) for i in temp_corner]
                temp_corners.append(temp_corner)
            temp_corners = np.array(temp_corners)
            temp_corners = torch.from_numpy(temp_corners)
            temp_corners = temp_corners  # / 128
            homography = torch.from_numpy(homography)
            # Convert 4*2 to 8*1
            homography = homography.view(-1, 8)
            # Reshape from 1*8 to 8
            homography = homography.squeeze(0)
            imgA = torch.from_numpy(imgA)
            imgB = torch.from_numpy(imgB)
            # Multiply by 255
            imgA = imgA  # * 255
            imgB = imgB  # * 255
            # Add axis
            imgA = imgA.unsqueeze(0)
            imgB = imgB.unsqueeze(0)
            patchAbatch.append(imgA)
            patchBbatch.append(imgB)
            img = img.permute(2, 0, 1)
            patchbatch.append(img)
            homographybatch.append(homography)
            imgbatch.append(img1)
            cornerbatch.append(temp_corners)
        # Convert the patch_a and patch_b batch to tensors by adding a batch dimension
        patchAbatch = torch.stack(patchAbatch)
        patchBbatch = torch.stack(patchBbatch)
        homographybatch = torch.stack(homographybatch)
        patchbatch = torch.stack(patchbatch)
        imgbatch = torch.stack(imgbatch)
        cornerbatch = torch.stack(cornerbatch)
        return patchAbatch, patchBbatch, patchbatch, imgbatch, homographybatch, cornerbatch


def PrettyPrint(epoch_num, DivTrain, MiniBatchSize, num_train_samples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of epochs Training will run for " + str(epoch_num))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(num_train_samples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    train_data_dir,
    TrainCoordinates,
    num_train_samples,
    ImageSize,
    epoch_num,
    MiniBatchSize,
    save_checkpoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    base_data_path,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    train_data_dir - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    num_train_samples - length(Train)
    ImageSize - Size of the image
    epoch_num - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    save_checkpoint - Save checkpoint every save_checkpoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    base_data_path - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    if ModelType == "Sup":
        model = SupNet((2, 128, 128), 8).to(device)
        Optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=4, gamma=0.1)
    else:
        model = Net((2, 128, 128), 8).to(device)
        Optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        start = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        start = 0
        print("New model initialized....")

    # dummy_input1 = torch.randn(MiniBatchSize, 1, 128, 128).to(device)  
    # dummy_input2 = torch.randn(MiniBatchSize, 1, 128, 128).to(device)
    # Writer.add_graph(model, (dummy_input1, dummy_input2))
    # Writer.flush()

    generate_batch_info = dict()

    # Base paths for data directories
    base_data_path = "../Data"
    train_data_dir = "GeneratedTrainData/"
    train_base_path = base_data_path + os.sep + "Train/"
    val_base_path = base_data_path + os.sep + "Val/"

    # Paths for training patches
    train_patch_a_path = base_data_path + os.sep + train_data_dir + "patchA/"
    train_patch_b_path = base_data_path + os.sep + train_data_dir + "patchB/"
    train_patch_a_list = os.listdir(train_patch_a_path)
    train_patch_b_list = os.listdir(train_patch_b_path)

    # Paths for validation patches
    val_data_dir = "GeneratedValData/"
    val_patch_a_path = base_data_path + os.sep + val_data_dir + "patchA/"
    val_patch_b_path = base_data_path + os.sep + val_data_dir + "patchB/"
    val_patch_a_list = os.listdir(val_patch_b_path)
    val_patch_b_list = os.listdir(val_patch_b_path)

    # Lists of base images for training and validation
    train_base_image_list = os.listdir(train_base_path)
    val_base_image_list = os.listdir(val_base_path)


    # Load homography and corner coordinate data for training and validation
    train_homography_labels = np.load("TxtFiles/homography_train_labels.npy", allow_pickle=True)
    val_homography_labels = np.load("TxtFiles/homography_val_labels.npy", allow_pickle=True)
    train_corner_coordinates = np.load("TxtFiles/train_corner_coordinates.npy", allow_pickle=True)
    val_corner_coordinates = np.load("TxtFiles/val_corner_coordinates.npy", allow_pickle=True)

    # Populate the batch generation info dictionary for training data
    generate_batch_info["train"] = {
        "patch_a": train_patch_a_path,
        "patch_b": train_patch_b_path,
        "patch_a_list": train_patch_a_list,
        "patch_b_list": train_patch_b_list,
        "corner_coordinates": train_corner_coordinates.item(),
        "homography": train_homography_labels.item(),
        "base_image_path": train_base_path,
        "base_image_path_list": train_base_image_list,
    }

    # Populate the batch generation info dictionary for validation data
    generate_batch_info["val"] = {
        "patch_a": val_patch_a_path,
        "patch_b": val_patch_b_path,
        "patch_a_list": val_patch_a_list,
        "patch_b_list": val_patch_b_list,
        "corner_coordinates": val_corner_coordinates.item(),
        "homography": val_homography_labels.item(),
        "base_image_path": val_base_path,
        "base_image_path_list": val_base_image_list,
    }

    # Number of training samples
    num_train_samples = len(train_patch_a_list)

    train_loss = []
    val_loss = []


    for epochs in tqdm(range(start, epoch_num)):
        num_iterations_per_epoch = int(num_train_samples / MiniBatchSize / DivTrain)
        train_batch = []
        val_batch = []
        for per_epoch_counter in tqdm(range(num_iterations_per_epoch)):
            train_batch = GenerateBatch(generate_batch_info["train"], MiniBatchSize, ModelType)
            val_batch = GenerateBatch(generate_batch_info["val"], MiniBatchSize,ModelType)
            
            train_batch = [x.to(device) for x in train_batch]
            val_batch = [x.to(device) for x in val_batch]

            batch_loss = model.training_step(train_batch)["loss"]

            Optimizer.zero_grad()
            batch_loss.backward()
            Optimizer.step()

            result = model.validation_step(val_batch)
            Writer.add_scalars(
                "LossEveryIter",
                {"train": batch_loss, "val": result["val_loss"]},
                epochs * num_iterations_per_epoch + per_epoch_counter,
            )
            train_loss.append(batch_loss.item())
            val_loss.append(result["val_loss"].item())
            Writer.flush()

            if per_epoch_counter % save_checkpoint == 0:
                SaveName = (
                    CheckPointPath
                    + str(epochs)
                    + "a"
                    + str(per_epoch_counter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": batch_loss,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            result = model.validation_step(val_batch)
            Writer.add_scalar(
                "LossEveryIter",
                result["val_loss"],
                epochs * num_iterations_per_epoch + per_epoch_counter,
            )
            Writer.flush()
        
        if ModelType == "Supervised":
            scheduler.step()
        avg_train_loss = round(sum(train_loss) / len(train_loss), 4)
        avg_val_loss = round(sum(val_loss) / len(val_loss), 4)
        SaveName = CheckPointPath + str(epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
        Writer.add_scalars(
            "LossPerEpoch", {"train": avg_train_loss, "val": avg_val_loss}, epochs
        )
        Writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epochs)
        Writer.flush()
        print("Epoch: ", epochs, "Train Loss: ", avg_train_loss)
        print("Epoch: ", epochs, "Val Loss: ", avg_val_loss)


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--base_data_path",
        default="../Data",
        help="Base path of images, Default:../Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Sup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--epoch_num",
        type=int,
        default=50,
        help="Number of epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    epoch_num = Args.epoch_num
    base_data_path = Args.base_data_path
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    if ModelType == "Sup":
        print("Using Supervised Model")
    elif ModelType == "Unsup":
        print("Using Unsupervised Model")
    else:
        print("ModelType must be Sup or Unsup, Defaulting to Sup")
        ModelType = "Sup"
    # Setup all needed parameters including file reading
    (
        train_data_dir,
        save_checkpoint,
        ImageSize,
        num_train_samples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(base_data_path, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(epoch_num, DivTrain, MiniBatchSize, num_train_samples, LatestFile)

    TrainOperation(
        train_data_dir,
        TrainCoordinates,
        num_train_samples,
        ImageSize,
        epoch_num,
        MiniBatchSize,
        save_checkpoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        base_data_path,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
