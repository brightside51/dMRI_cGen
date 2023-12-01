# Library Imports
import sys
import argparse
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Functionality Import
from pathlib import Path
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential

##############################################################################################
# ------------------------------- Voxel-Wise Fixed cglVNN Build ------------------------------
##############################################################################################

# Fixed Conditional Generative Linear Voxel Net Model Class
class fcglVNN(keras.Model):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
        #in_channels: int = 500,                 # Number of Input Parameter Settings
        #num_labels: int = 5,                    # Number of Training Parameters / Input Channels
        #num_hidden: int = 3,                    # Number of NN Hidden Layers
        #var_hidden: int = 128                   # Deviance / Expansion of Hidden Layers
    ):

        # Class Variable Logging
        super().__init__()
        self.in_params = settings.in_params; self.num_labels = settings.num_labels
        self.num_hidden = settings.num_hidden; self.var_hidden = settings.var_hidden
        self.net = Sequential(); out_neuron = self.var_hidden

        # Neural Network Architecture Definition
        self.net.add(Dense( input_dim = self.in_channels + self.num_labels,
                            units = self.var_hidden))
        for i in range(self.num_hidden + 1):
            if i == self.num_hidden: out_neuron = 1
            else: out_neuron = int(out_neuron / 2)
            self.net.add(   LeakyReLU(alpha = 0.2))
            self.net.add(   Dense(units = out_neuron))
    
    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def call(
        self,
        X_real: np.ndarray,
        y_target: np.ndarray
    ):  return self.net(tf.keras.layers.concatenate([X_real, y_target], axis = 1))

##############################################################################################
# ------------------------------- Voxel-Wise Fixed cglVNN Build ------------------------------
##############################################################################################