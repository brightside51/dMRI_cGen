# Library Imports
import argparse
import numpy as np
import torch
import torch.nn as nn

##############################################################################################
# ------------------------------- Voxel-Wise Fixed cglVNN Build ------------------------------
##############################################################################################

# Fixed Conditional Generative Linear Voxel Net Model Class
class fcglVNN(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):
        
        # Class Variable Logging
        super().__init__(); self.settings = settings
        self.net = []; self.arch = []
        self.arch.insert(0, self.settings.in_channels + self.settings.num_labels)

        # Neural Network Architecture Definition
        var_hidden = int((self.settings.top_hidden - self.settings.bottom_hidden) / self.settings.num_hidden)
        for i in range(1, self.settings.num_hidden + 1):
            if i == 1: self.arch.insert(i, self.settings.bottom_hidden + var_hidden)
            else: self.arch.insert(i, self.arch[i - 1] + var_hidden)
            self.main_block(self.arch[i], self.arch[i - 1])
        for i in range(self.settings.num_hidden + 1, (2 * self.settings.num_hidden) + 1):
            self.arch.insert(i, self.arch[i - 1] - var_hidden)
            self.main_block(self.arch[i], self.arch[i - 1])
        self.net.append(nn.Linear(in_features = self.settings.bottom_hidden, out_features = 1))
        self.net = nn.Sequential(*self.net)

    # Block Architecture Definition Functionality
    def main_block(self, out_channels: int, in_channels: int):
        self.net.append(
            nn.Sequential(  nn.Linear(      in_features = in_channels,
                                            out_features = out_channels),
                            #nn.SyncBatchNorm(num_features = out_channels),
                            nn.BatchNorm1d( num_features = out_channels),
                            nn.ReLU()))
                            #nn.LeakyReLU(   negative_slope = 0.2)))
    
    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def forward(
        self,
        input: np.ndarray or torch.Tensor
    ):  return self.net(input)