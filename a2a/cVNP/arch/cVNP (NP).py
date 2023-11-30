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
        super().__init__(); self.settings = settings
        self.encoder = []; self.decoder = []; self.arch = []
        self.arch.insert(0, 1 + self.settings.num_labels)

        # Encoder Architecture Definition
        for i in range(self.settings.num_hidden):
            self.arch.insert(i + 1, int(self.settings.var_hidden /\
                            (2 ** (self.settings.num_hidden - i - 1))))
            self.encoder.append(nn.Sequential(
                            nn.Linear(      in_features = self.arch[i],
                                            out_features = self.arch[i + 1]),
                            nn.BatchNorm1d( num_features = self.arch[i + 1]),
                            nn.ReLU()))
        self.encoder = nn.Sequential(*self.encoder)
        
        # --------------------------------------------------------------------------------------------
        
        # Decoder Architecture Definition
        self.arch.insert(len(self.arch), self.settings.var_hidden + self.settings.num_labels)
        for i in range(self.settings.num_hidden):
            i += self.settings.num_hidden + 1
            self.arch.insert(i + 1, int(self.settings.var_hidden /\
                            (2 ** (i - self.settings.num_hidden - 1))))
            self.decoder.append(nn.Sequential(
                            nn.Linear(      in_features = self.arch[i],
                                            out_features = self.arch[i + 1]),
                            nn.BatchNorm1d( num_features = self.arch[i + 1]),
                            nn.ReLU()))
        self.decoder.append(nn.Linear(  in_features = self.arch[-1],
                                        out_features = 1))
        self.decoder = nn.Sequential(*self.decoder)            

    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def forward(
        self,
        X_train: np.ndarray or torch.Tensor,
        y_train: np.ndarray or torch.Tensor,
        y_target: np.ndarray or torch.Tensor
    ):  
        
        # Batch Data Handling
        #X_train = X_train.reshape(X_train.shape[0], 1).to(torch.float32)
        #y_train = y_train.to(torch.float32)
        #y_target = y_target.to(torch.float32)

        # Model Network Application
        X_train = X_train.reshape(X_train.shape[0], 1)
        z = self.encoder(torch.cat((X_train, y_train), dim = 1))    # Encoder Feedthrough
        #z = z.repeat(y_target.shape[0], 1)                         # Aggregation
        return self.decoder(torch.cat((z, y_target), dim = 1))      # Decoder Feedthrough