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
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

##############################################################################################
# ------------------------------- Voxel-Wise Fixed cglVNN Build ------------------------------
##############################################################################################

# Fixed Conditional Generative Linear Voxel Net Model Class (Fixed)
class cglVNN(tf.keras.Model):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser
    ):
        
        # Class Variable Logging
        super().__init__(); self.settings = settings
        self.net = Sequential(); self.arch = []
        self.arch.insert(0, 1 + (2 * self.settings.num_labels))

        # Neural Network Architecture Definition
        var_hidden = int((self.settings.top_hidden - self.settings.bottom_hidden) / self.settings.num_hidden)
        for i in range(1, self.settings.num_hidden + 1):
            if i == 1: self.arch.insert(i, self.settings.bottom_hidden * 2)
            else: self.arch.insert(i, int(self.arch[i - 1] * 2))
            self.main_block(self.arch[i], self.arch[i - 1])
        for i in range(self.settings.num_hidden + 1, (2 * self.settings.num_hidden) + 1):
            self.arch.insert(i, int(self.arch[i - 1] / 2))
            self.main_block(self.arch[i], self.arch[i - 1])
        self.net.add(Dense(input_dim = self.settings.bottom_hidden, units = 1))

    # Block Architecture Definition Functionality
    def main_block(self, out_channels: int, in_channels: int = None):
        if in_channels is not None:
            self.net.add(   Dense(  input_dim = in_channels,
                                    units = out_channels))
        else: self.net.add( Dense(  units = out_channels))
        self.net.add(   BatchNormalization())
        self.net.add(   LeakyReLU(  alpha = 0.2))
    
    # --------------------------------------------------------------------------------------------

    # Neural Network Application Function
    def call(
        self,
        input: np.ndarray or tf.Tensor
    ):  return self.net(input)