# Library Imports
import argparse
import numpy as np
import torch
import torch.nn as nn

##############################################################################################
# ------------------------------------ CVAE Decoder Build ------------------------------------
##############################################################################################

# CVAE Decoder Main Block Class
class DecoderBlock(nn.Module):

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
        super().__init__(); block = []
        block.append(nn.Sequential(
                        nn.ConvTranspose2d( in_channels = in_channels, out_channels = out_channels,
                                            kernel_size = kernel_size, stride = stride,
                                            padding = padding, output_padding = 1, bias = False),
                        nn.BatchNorm2d(     num_features = out_channels, momentum = momentum), nn.ReLU()))
        self.block = nn.Sequential(*block)

    # Block Application Function
    def forward(self, X): return self.block(X)
                            
# --------------------------------------------------------------------------------------------

# CVAE Decoder Model Class
class Decoder(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):

        # CVAE Decoder Linear Architecture Definition
        super().__init__(); self.settings = settings
        self.img_shape = int(np.ceil(self.settings.img_shape / (2 ** (self.settings.num_hidden))))
        out_channels = int((self.settings.dim_latent / 2) * (2 ** (self.settings.num_hidden - 1)) * (self.img_shape ** 2))
        self.decoder_fc = nn.Sequential(
                                nn.Linear(      in_features = self.settings.dim_latent + self.settings.num_labels,
                                                out_features = out_channels, bias = False),
                                nn.BatchNorm1d( num_features = out_channels, momentum = 0.9),
                                nn.ReLU(inplace = True))

        # CVAE Decoder Convolutional Architecture Definition
        in_channels = int(out_channels / (self.img_shape ** 2))
        decoder_conv = []; out_channels = self.settings.dim_latent * 2
        for i in range(self.settings.num_hidden):
            #print(f"{in_channels} -> {out_channels}")
            decoder_conv.append(DecoderBlock(   in_channels = in_channels,
                                                out_channels = out_channels,
                                                kernel_size = self.settings.kernel_size,
                                                padding = self.settings.padding))
            in_channels = out_channels; out_channels = int(out_channels / 2)
        decoder_conv.append(nn.Sequential(
                                    nn.Conv2d(  in_channels = in_channels, out_channels = 1,
                                                kernel_size = self.settings.kernel_size,
                                                padding = self.settings.padding), nn.Tanh()))
        self.decoder_conv = nn.Sequential(*decoder_conv)
                
    # --------------------------------------------------------------------------------------------

    # Reparametrization Trick Functionality
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)
    
    # Kullback-Leibler Divergence Computation Functionality
    def kl_loss(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # --------------------------------------------------------------------------------------------

    # CVAE Decoder Application Function
    def forward(
        self,
        z: np.ndarray or torch.Tensor,
        y_target: np.ndarray or torch.Tensor
    ): 
        out = torch.cat((z, y_target), dim = 1)     # Inclusion of Target Labels
        out = self.decoder_fc(out)                  # Linear Section Application
        out = out.view( len(out), -1,
                        self.img_shape,
                        self.img_shape)             # Output Dimensionalization
        return self.decoder_conv(out)               # Convolutional Section Application
