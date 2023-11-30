# Library Imports
import argparse
import numpy as np
import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

##############################################################################################
# ---------------------------------- GAN Discriminator Build ---------------------------------
##############################################################################################

# fcgCVAE Model Class
class fcgCVAE(nn.Module):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):
        super().__init__(); self.settings = settings
        self.encoder = Encoder(self.settings)
        self.decoder = Decoder(self.settings)

    # CVAE Model Application Function
    def forward(
        self,
        X_train: np.ndarray or torch.Tensor,
        y_target: np.ndarray or torch.Tensor
    ):
        mu, logvar = self.encoder(X_train)              # Encoder Model Application
        z = fcgCVAE.reparam(mu, logvar)                 # Reparametrization Trick
        kl_loss = fcgCVAE.kl_loss(mu, logvar)           # Kullback-Leibler Loss Computation
        return self.decoder(z, y_target),kl_loss        # Decoder Model Application

    # --------------------------------------------------------------------------------------------

    # Reparametrization Trick Functionality
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)
    
    # Kullback-Leibler Divergence Computation Functionality
    def kl_loss(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    