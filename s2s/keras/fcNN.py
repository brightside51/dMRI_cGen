# Library Imports
import sys
import argparse
import numpy as np
import pandas as pd
import keras
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Functionality Import
from pathlib import Path
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential

# Full Voxel-Wise CVAE Model Class Importing
#sys.path.append('../Model Builds')
#from Encoder import Encoder
#from Decoder import Decoder

##############################################################################################
# ----------------------------------- Voxel-Wise fcNN Build ----------------------------------
##############################################################################################

# Linear Fully Connected Neural Network Model Class
class fcNN(keras.Model):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
        #in_params: int = 500,                   # Number of Input Parameter Settings
        #out_params: int = 1344,                 # Number of Output Parameter Settings
        #num_hidden: int = 2                     # Number of NN Hidden Layers
    ):

        # Class Variable Logging
        super().__init__()
        self.in_params = settings.in_params
        self.out_params = settings.out_params
        self.num_hidden = settings.num_hidden
        assert(self.out_params > self.in_params),"ERROR: Neural Network wrongly built!"
        self.net = Sequential(); num_neuron = self.in_params

        # Neural Network Architecture Definition
        num_fc = int(np.floor((self.out_params - self.in_params) / (self.num_hidden + 1)))
        for i in range(self.num_hidden + 1):
            if i == 0:
                self.net.add(   Dense(input_dim = num_neuron, units = num_neuron + num_fc))
                self.net.add(   LeakyReLU(alpha = 0.2))
            elif i == self.num_hidden:
                num_fc += 1; self.net.add(  Dense(units = num_neuron + num_fc))
            else:
                self.net.add(   Dense(units = num_neuron + num_fc))
                self.net.add(   LeakyReLU(alpha = 0.2))
            num_neuron += num_fc
        assert(num_neuron == self.out_params), "ERROR: Neural Network wrongly built!"
    
    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def call(
        self,
        X: np.ndarray,
    ):  return self.net(X)