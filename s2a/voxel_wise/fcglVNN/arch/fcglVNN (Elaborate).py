# Library Imports
import sys
import argparse
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# Functionality Import
from pathlib import Path
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential

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

        # Metric Function Initialization
        self.total_loss_metric = tf.keras.metrics.Mean(name = 'Loss')
        self.train_loss_metric = tf.keras.metrics.Mean(name = 'Training Loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name = 'Validation Loss')
        
        # Reconstruction Parameter Initialization
        self.num_train_recon = int((self.settings.param_recon_train * self.settings.num_train_params) / 100)
        self.num_val_recon = int((self.settings.param_recon_train * (1344 - self.settings.num_train_params)) / 100)
        self.num_recon = self.num_train_recon + self.num_val_recon; self.batch_idx = 0
    
    # --------------------------------------------------------------------------------------------

    # Custom Object Configuration Function
    #@classmethod
    #def from_config(cls, config): return cls(**config)
    #def get_config(self): return {'Loss': self.total_loss.numpy()}

    # Neural Network Application Function
    def call(
        self,
        input: np.ndarray or tf.Tensor
        #X_train: np.ndarray or tf.Tensor,
        #y_target: np.ndarray or tf.Tensor
    ):  return self.net(input)

    ##############################################################################################
    # ------------------------------------- Training Script --------------------------------------
    ##############################################################################################

    # Functionality called upon the End of Training Epoch
    def on_epoch_end(self, epoch, logs = None): self.batch_idx = 0
        
    # Training Step / Batch Loop
    def train_step(self, data):        

        # Voxel Reconstruction Loop
        input, X_gt = data#[0]
        with tf.GradientTape() as tape:

            # Forward Propagation
            X_target = self(input, training = True)     # Predicted Target Voxel Intensity
            loss = self.compiled_loss(X_gt, X_target,   # Loss Computation using
                regularization_losses = self.losses)    # Compiled Loss Function (MSE)

        # BackPropagation
        var = self.trainable_variables                          # Trainable Variables Fetching
        grad = tape.gradient(loss, var)                         # Gradient Computation
        self.optimizer.apply_gradients(zip(grad, var))          # Optimizer Weight Update

        # Loss Organization
        self.total_loss_metric.update_state(loss)               # Total Epoch Loss Update
        if self.batch_idx % self.num_recon < self.num_train_recon:
            self.train_loss_metric.update_state(loss)           # Training Parameter Loss Update
        else: self.val_loss_metric.update_state(loss)           # Validation Parameter Loss Update
        self.batch_idx += 1
        
        return {'Loss': self.total_loss_metric.result(),
                'Training Loss': self.train_loss_metric.result(),
                'Validation Loss': self.val_loss_metric.result()}

    # Loss Metrics Reset
    @property
    def metrics(self): return [self.total_loss_metric, self.train_loss_metric, self.val_loss_metric]
