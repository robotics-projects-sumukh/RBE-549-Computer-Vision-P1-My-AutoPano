"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
from kornia.geometry.transform import warp_perspective

# Don't generate pyc codes
sys.dont_write_bytecode = True


def direct_linear_transform(corners, corners_hat):
    B = corners.shape[0]
    x = corners[..., 0]   
    y = corners[..., 1]   
    x_hat = corners_hat[..., 0]  
    y_hat = corners_hat[..., 1]
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    row1 = torch.stack([-x, -y, -ones, zeros, zeros, zeros, x * x_hat, y * x_hat, x_hat], dim=2)
    row2 = torch.stack([zeros, zeros, zeros, -x, -y, -ones, x * y_hat, y * y_hat, y_hat], dim=2)
    A = torch.cat([row1, row2], dim=1)
    U, S, Vh = torch.linalg.svd(A)
    h = Vh[:, -1, :] 
    h = h.view(B, 3, 3)
    h = h / h[:, 2:3, 2:3]
    return h

def LossFn(delta, img_a, patch_b, corners):
    corners_hat = corners + delta
    corners = corners - corners[:, 0].view(-1, 1, 2)
    h = direct_linear_transform(corners, corners_hat)
    h_inv = torch.inverse(h)
    patch_b_hat = warp_perspective(img_a, h_inv, (128, 128))
    return F.l1_loss(patch_b_hat, patch_b)


class HomographyModel(nn.Module):
    def training_step(self, batch):
        patcha, patchb, patch, imgA, gt, corners = batch
        delta = self(patcha, patchb)
        loss = LossFn(delta, imgA, patchb, corners)
        return {"loss": loss}

    def validation_step(self, batch):
        patcha, patchb, patch, imgA, gt, corners = batch
        delta = self(patcha, patchb)
        loss = LossFn(delta, imgA, patchb, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Block(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
        super(Block, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Net(HomographyModel):
    def __init__(self,InputSize, OutputSize, batch_norm=False):
        super(Net, self).__init__()
        self.cnn = nn.Sequential(
            Block(2, 64, batch_norm),
            Block(64, 64, batch_norm),
            Block(64, 128, batch_norm),
            Block(128, 128, batch_norm, pool=False),
        )
        self.fc = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 4 * 2),
        )

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)  # combine two images in channel dimension
        x = self.cnn(x)
        x = self.fc(x)
        delta = x.view(-1, 4, 2)
        return delta
    