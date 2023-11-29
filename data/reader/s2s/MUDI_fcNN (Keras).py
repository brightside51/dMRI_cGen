# Library Imports
import os
import pickle
import random
import numpy as np
import argparse
import pandas as pd
import keras
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py

# Functionality Import
from pathlib import Path
from sklearn.model_selection import train_test_split
from nilearn.image import load_img
from nilearn.masking import unmask
from scipy.ndimage.interpolation import rotate
from sklearn.preprocessing import StandardScaler
from ipywidgets import interactive, IntSlider
from tabulate import tabulate
from alive_progress import alive_bar

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# 1D MUDI Dataset Initialization Class
class MUDI_fcNN(keras.utils.Sequence):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        subject: int or list
    ):
        
        # Parameter & Patient Index Value Access
        super(MUDI_fcNN).__init__()
        self.settings = settings

        # Vertical Splitting (Patient Selection)
        self.idxv = pd.read_csv(self.settings.info_filepath,            # List of Index Values ...
                                index_col = 0).to_numpy()               # ... pertaining to each Patient
        self.idxv = self.idxv[np.isin(self.idxv[:, 1], subject), 0]     # Patient-Specific Index Values
        self.idxv_list = np.arange(len(self.idxv))

        # Horizontal Splitting (Parameter Selection)
        self.idxh_train_filepath = Path(f"{self.settings.save_folderpath}/1D Training Labels (V{self.settings.version}).txt")    # Filepath for Selected Training Labels
        self.idxh_val_filepath = Path(f"{self.settings.save_folderpath}/1D Validation Labels (V{self.settings.version}).txt")    # Filepath for Selected Validation Labels
        self.idxh_train = np.sort(np.loadtxt(self.idxh_train_filepath)).astype(int)
        self.idxh_val = np.sort(np.loadtxt(self.idxh_val_filepath)).astype(int)
        #self.msk = np.logical_not(np.isin(np.arange(1344), self.idxh_train))
        
        # Parameter Value Initialization & Selection
        self.params = pd.read_excel(self.settings.param_filepath)                                       # List of Dataset's Parameters
        self.num_labels = self.settings.num_labels
        if self.settings.gradient_coord:
            self.params = self.params.drop(columns = ['Gradient theta', 'Gradient phi'])                # Choosing of Gradient Orientation
        else: self.params = self.params.drop(columns = ['Gradient x', 'Gradient y', 'Gradient z'])      # from 3D Cartesian or Polar Referential
        assert(self.params.shape[1] == self.num_labels), "ERROR: Labels wrongly Deleted!"
        if self.settings.label_norm:                                                                    # Control Boolean Value for the Normalization of Labels
            self.scaler = StandardScaler()
            self.params = pd.DataFrame( self.scaler.fit_transform(self.params),
                                        columns = self.params.columns)
    
    # --------------------------------------------------------------------------------------------

    # DataLoader Length / No. Batches Computation Functionality
    def __len__(self): return int(np.ceil(len(self.idxv) / float(self.settings.batch_size)))

    # Single-Batch Generation Functionality
    def __getitem__(self, idxv):
        idxv = self.idxv_list[idxv * self.settings.batch_size : (idxv + 1) * self.settings.batch_size]
        idxv = [self.idxv[k] for k in idxv]
        X_train, X = self.__data_generation(idxv)
        return X_train, X
    
    # --------------------------------------------------------------------------------------------

    # End of Epoch Shuffling Functionality
    def on_epoch_end(self):
        self.idxv_list = np.arange(len(self.idxv))
        if self.settings.sample_shuffle: np.random.shuffle(self.idxv_list)

    # Data Generation Functionality
    def __data_generation(self, idxv):
        data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        X = data[idxv, :]; X_train = X[:, self.idxh_train]
        return X_train, X
