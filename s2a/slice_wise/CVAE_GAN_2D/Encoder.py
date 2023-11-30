# Library Imports
import argparse
import numpy as np
import torch
import torch.nn as nn

##############################################################################################
# ------------------------------------ CVAE Encoder Build ------------------------------------
##############################################################################################

# CVAE Encoder Main Block Class
class EncoderBlock(nn.Module):

    # Block Constructor / Initialization Function
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        momentum: float = 0.9
    ):

        # Class Variable Logging
        super().__init__()
        self.block_conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                    kernel_size = kernel_size, stride = stride, padding = padding)
        self.block_rest = nn.Sequential(
                    nn.BatchNorm2d( num_features = out_channels, momentum = momentum),
                    nn.ReLU(        inplace = False))

    # Block Application Function
    def forward(self, X, recon: bool = False):
        layer_rep = self.block_conv(X)
        out = self.block_rest(layer_rep)
        if recon: return out, layer_rep
        else: return out

# --------------------------------------------------------------------------------------------

# CVAE Encoder Model Class
class Encoder(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):
        
        # CVAE Encoder Convolutional Architecture Definition
        super().__init__(); self.settings = settings; encoder_conv = []
        in_channels = 1; out_channels = int(self.settings.dim_latent / 2)
        for i in range(self.settings.num_hidden):
            #print(f"{in_channels} -> {out_channels}")
            encoder_conv.append(EncoderBlock(   in_channels = in_channels,
                                                out_channels = out_channels,
                                                kernel_size = self.settings.kernel_size,
                                                padding = self.settings.padding))
            in_channels = out_channels; out_channels *= 2
        self.encoder_conv = nn.Sequential(*encoder_conv)

        # CVAE Encoder Linear Architecture Definition
        img_shape = np.ceil(self.settings.img_shape / (2 ** (self.settings.num_hidden)))
        in_channels = int(in_channels * (img_shape ** 2))
        self.encoder_fc = nn.Sequential(
                                nn.Linear(      in_features = in_channels,
                                                out_features = self.settings.dim_hidden, bias = False),
                                nn.BatchNorm1d( num_features = self.settings.dim_hidden, momentum = 0.9),
                                nn.ReLU(        inplace = True))
        self.encoder_mu = nn.Linear(            in_features = self.settings.dim_hidden + self.settings.num_labels,
                                                out_features = self.settings.dim_latent)
        self.encoder_logvar = nn.Linear(        in_features = self.settings.dim_hidden + self.settings.num_labels,
                                                out_features = self.settings.dim_latent)

    # --------------------------------------------------------------------------------------------
    
    # CVAE Encoder Application Function
    def forward(
        self,
        X_train: np.ndarray or torch.Tensor,
        y_train: np.ndarray or torch.Tensor
    ): 
        out = self.encoder_conv(X_train)            # Convolutional Section Application
        out = out.view(len(out), -1)                # Output Linearization
        out = self.encoder_fc(out)                  # Linear Section Application
        out = torch.cat((out, y_train), dim = 1)    # Inclusion of Training Labels
        mu = self.encoder_mu(out)                   # Mean Computation
        logvar = self.encoder_logvar(out)           # Log Variance Computation
        return mu, logvar
