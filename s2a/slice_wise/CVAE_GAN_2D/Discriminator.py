# Library Imports
import argparse
import numpy as np
import torch
import torch.nn as nn
from Encoder import EncoderBlock

##############################################################################################
# ---------------------------------- GAN Discriminator Build ---------------------------------
##############################################################################################

# GAN Discriminator Model Class
class Discriminator(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):
        
        # GAN Discriminator Convolutional Architecture Definition #1
        super().__init__(); self.settings = settings
        self.discriminator_conv = nn.ModuleList()
        out_channels = self.settings.dim_latent // 4
        self.discriminator_conv.append(nn.Sequential(
                    nn.Conv2d(  in_channels = 1, out_channels = out_channels,
                                kernel_size = self.settings.kernel_size,
                                padding = self.settings.padding),
                    nn.ReLU(    inplace = True)))

        # GAN Discriminator Convolutional Architecture Definition #2
        for i in range(self.settings.num_hidden):
            #print(f"{in_channels} -> {out_channels}")
            in_channels = out_channels
            if i == 0: out_channels *= 4
            else: out_channels *= 2
            if i == self.settings.num_hidden - 1: out_channels //= 2
            self.discriminator_conv.append(
                EncoderBlock(   in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = self.settings.kernel_size,
                                padding = self.settings.padding))
            
        # GAN Discriminator Linear Architecture Definition
        self.img_shape = int(np.ceil(self.settings.img_shape / (2 ** (self.settings.num_hidden))))
        self.discriminator_fc = nn.Sequential(
                nn.Linear(      in_features = out_channels * (self.img_shape ** 2),
                                out_features = self.settings.dim_hidden // 2, bias = False),
                nn.BatchNorm1d( num_features = self.settings.dim_hidden // 2,
                                momentum = 0.9), nn.ReLU(inplace = True))
        self.discriminator_pred = nn.Sequential(
                    nn.Linear(  in_features = self.settings.dim_hidden // 2,
                                out_features = 1), nn.Sigmoid())
        self.discriminator_label = nn.Sequential(
                    nn.Linear(  in_features = self.settings.dim_hidden // 2,
                                out_features = self.settings.num_labels), nn.LogSoftmax())

    # --------------------------------------------------------------------------------------------

    # GAN Discriminator Model Application Function
    def forward(
        self,
        X_gt: np.ndarray or torch.Tensor,
        X_target: np.ndarray or torch.Tensor,
        recon: bool = False
    ):

        # GAN Discriminator Convolutional Architecture Application
        input = torch.cat((X_gt, X_target), dim = 0)
        if recon:
            for i, layer in enumerate(self.discriminator_conv):
                if i != self.settings.recon_level: input = layer(input)
                else:
                    input, recon_pred = layer(input, recon = True)          # Layer Representations Output
                    recon_pred = recon_pred.view(len(recon_pred), -1)       # for both Input & Reconstructed Image
                    return nn.functional.sigmoid(recon_pred)
        else:
            for i, layer in enumerate(self.discriminator_conv): input = layer(input)
        
        # GAN Discriminator Linear Architecture Application
        out = input.view(len(input), -1)
        out = self.discriminator_fc(out)
        auth_pred = self.discriminator_pred(out)        # Real vs Fake Prediction
        #label_pred = self.discriminator_label(out)      # Label Value Prediction
        return auth_pred
