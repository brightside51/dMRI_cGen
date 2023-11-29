# Library Imports
import argparse
import numpy as np
import torch
import torch.nn as nn

##############################################################################################
# ------------------------------- Voxel-Wise Fixed cglVNN Build ------------------------------
##############################################################################################

# Fixed Conditional Generative Linear Voxel Net Model Class
class fcNN(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):
        
        # Class Variable Logging
        super().__init__(); self.settings = settings
        self.net = []; self.arch = []
        self.arch.insert(0, self.settings.in_channels)

        # Neural Network Architecture Definition
        var_hidden = int((self.settings.out_channels - self.settings.in_channels) / (self.settings.num_hidden + 1))
        for i in range(1, self.settings.num_hidden + 2):
            if i == self.settings.num_hidden + 1:
                self.arch.insert(i, self.settings.out_channels)
                self.net.append(    nn.Linear(      in_features = self.arch[i - 1],
                                                    out_features = self.arch[i]))
            else:
                self.arch.insert(i, self.arch[i - 1] + var_hidden)
                self.net.append(
                    nn.Sequential(  nn.Linear(      in_features = self.arch[i - 1],
                                                    out_features = self.arch[i]),
                                    nn.BatchNorm1d( num_features = self.arch[i]),
                                    nn.ReLU()))
        self.net = nn.Sequential(*self.net)
    
    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def forward(
        self,
        input: np.ndarray or torch.Tensor
    ):  return self.net(input)