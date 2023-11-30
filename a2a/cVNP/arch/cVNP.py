# Library Imports
import argparse
import numpy as np
import torch
import torch.nn as nn

##############################################################################################
# ----------------------------------- Voxel-Wise CVNP Build ----------------------------------
##############################################################################################

# Fixed Conditional Generative Linear Voxel Net Model Class
class cVNP(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):
        
        # Class Variable Logging
        super().__init__(); self.settings = settings; self.net = []
        self.net.append(nn.Sequential(
                            nn.Linear(      in_features = 1 + (2 * self.settings.num_labels),
                                            out_features = 8),
                            nn.BatchNorm1d( num_features = 8),
                            nn.ReLU()))
        self.net.append(    nn.Linear(  in_features = 8,
                                        out_features = 1))
        self.net = nn.Sequential(*self.net)
     
    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def forward(
        self,
        X_train: np.ndarray or torch.Tensor,
        y_train: np.ndarray or torch.Tensor,
        y_target: np.ndarray or torch.Tensor
    ):  
        
        # Model Network Application
        X_train = X_train.reshape(X_train.shape[0], 1)
        return self.net(torch.cat((X_train, y_train, y_target), dim = 1))