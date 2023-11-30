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
from sklearn.preprocessing import StandardScaler

##############################################################################################
# ------------------------------- Voxel-Wise Fixed cglVNN Build ------------------------------
##############################################################################################

# Fixed Conditional Generative Linear Voxel Net Model Class
class fcglVNN(tf.keras.Model):

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
        super().__init__(); self.settings = settings
        self.net = Sequential(); out_neuron = self.settings.var_hidden

        # Neural Network Architecture Definition
        self.net.add(Dense( input_dim = self.settings.in_channels + self.settings.num_labels,
                            units = self.settings.var_hidden))
        for i in range(self.settings.num_hidden + 1):
            if i == self.settings.num_hidden: out_neuron = 1
            else: out_neuron = int(out_neuron / 2)
            self.net.add(   LeakyReLU(alpha = 0.2))
            self.net.add(   Dense(units = out_neuron))
    
    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def call(
        self,
        input: np.ndarray or tf.Tensor
        #X_train: np.ndarray or tf.Tensor,
        #y_target: np.ndarray or tf.Tensor
    ):  return self.net(input)