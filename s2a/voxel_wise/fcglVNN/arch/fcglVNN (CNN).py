# Library Imports
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Functionality Import
from pathlib import Path
from torchsummary import summary

##############################################################################################
# ------------------------------- Voxel-Wise Fixed cglVNN Build ------------------------------
##############################################################################################

# Fixed Conditional Generative Linear Voxel Net Model Class
class fcglVNN (nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        num_labels: int = 5,                    # Number of Input Parameter Settings
        in_channels: int = 500,                 # Number of Training Parameters / Input Channels
        num_hidden: int = 3,                    # Number of NN Hidden Layers
        var_hidden: int = 128,                  # Deviance / Expansion of Hidden Layers
    ):

        # Class Variable Logging
        super(fcglVNN, self).__init__()
        self.num_labels = num_labels; self.in_channels = in_channels
        self.num_hidden = num_hidden; self.var_hidden = var_hidden
        #var_neuron = int(np.floor(self.var_hidden / self.num_hidden))

        # Convolutional Neural Network Section Architecture Definition
        cBlock = []; lBlock = []; in_neuron = 1
        for i in reversed(range(num_hidden)):
            out_neuron = int(var_hidden / (2 ** i))
            cBlock.append( nn.Sequential(
                            nn.Conv1d(in_neuron, out_neuron, stride = 1,
                                      kernel_size = 3, padding = 1),
                            nn.LeakyReLU(inplace = True)))
            in_neuron = out_neuron

        # Linear Neural Network Section Architecture Definition
        lBlock.append(     nn.Sequential(
                            nn.Linear(var_hidden * (self.in_channels + self.num_labels), var_hidden * 2),
                            nn.LeakyReLU(inplace = True),
                            nn.Linear(var_hidden * 2, 1)))
        self.cBlock = nn.Sequential(*cBlock); self.lBlock = nn.Sequential(*lBlock)

    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def forward(
        self,
        X_real: np.ndarray or torch.Tensor,     # 1D Voxel Input
        y_target: np.ndarray or torch.Tensor    # 1D Target Label Input
    ):  

        # Neural Network Feed Forward Process
        out = torch.cat([X_real, y_target], dim = 1)
        out = out.unsqueeze(1)
        out = self.cBlock(out)
        out = out.view(out.size(0), -1)
        return self.lBlock(out)
