# Library Imports
import os
import pickle
import random
import numpy as np
import argparse
import pandas as pd
import keras
import tensorflow as tf
import torch
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
class MUDI_cglVNN(keras.utils.Sequence):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        subject: int or list,
        training: bool = True
    ):
        
        # Parameter & Patient Index Value Access
        super(MUDI_cglVNN).__init__()
        self.settings = settings

        # Vertical Splitting (Patient Selection)
        self.idxv = pd.read_csv(self.settings.info_filepath,            # List of Index Values ...
                                index_col = 0).to_numpy()               # ... pertaining to each Patient
        self.idxv = self.idxv[np.isin(self.idxv[:, 1], subject), 0]     # Patient-Specific Index Values
        self.idxv_set = np.arange(len(self.idxv))

        # Horizontal Splitting (Parameter Selection)
        idxh_train_filepath = Path(f"{self.settings.datasave_folderpath}/1D Training Labels (V{self.settings.data_version}).txt")    # Filepath for Selected Training Labels
        idxh_val_filepath = Path(f"{self.settings.datasave_folderpath}/1D Validation Labels (V{self.settings.data_version}).txt")    # Filepath for Selected Validation Labels
        self.idxh_train = np.sort(np.loadtxt(idxh_train_filepath)).astype(int); self.num_train_params = len(self.idxh_train)
        self.idxh_val = np.sort(np.loadtxt(idxh_val_filepath)).astype(int); self.num_val_params = len(self.idxh_val)
        assert(self.num_train_params == self.settings.num_train_params), "ERROR: Model wrongly Built!"
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
        
        # Label Normalizer / Scaler Saving
        scaler_filepath = Path(f"{self.settings.datasave_folderpath}/1D Label Scaler (V{self.settings.data_version}).pkl")
        if not scaler_filepath.exists(): torch.save(self.scaler, scaler_filepath)
        
        # Random Selection of Parameters for Training Loop Reconstruction
        if training:
            self.param_recon = self.settings.param_recon_train
            self.batch_size = self.settings.batch_size
        else:
            self.param_recon = self.settings.param_recon_full
            self.batch_size = len(self.idxv)
        self.num_train_recon = int((self.param_recon * self.num_train_params) / 100)
        self.num_val_recon = int((self.param_recon * self.num_val_params) / 100)
        self.num_recon = self.num_train_recon + self.num_val_recon
        #self.batch_size = self.settings.batch_size * self.num_recon
        self.idxh_recon = np.hstack((   self.idxh_train[np.sort(np.random.choice(self.num_train_params,
                                                                self.num_train_recon, replace = False))],
                                        self.idxh_val[np.sort(np.random.choice(self.num_val_params,
                                                                self.num_val_recon, replace = False))]))
    
    # --------------------------------------------------------------------------------------------

    # DataLoader Length / No. Batches Computation Functionality
    def __len__(self): return int(np.ceil(len(self.idxv) / self.batch_size)) * self.num_recon * self.settings.num_train_params

    # Single-Batch Generation Functionality
    def __getitem__(self, idx):

        # Batch Vertical/Patient Indexing
        idxv = int(idx // (self.num_recon * self.settings.num_train_params))
        idxv = self.idxv_set[   idxv * self.batch_size :
                                (idxv + 1) * self.batch_size]
        idxv = [self.idxv[k] for k in idxv]

        # Batch Horizontal/Parameter Indexing
        idxh_train = int((idx // self.num_recon) % self.settings.num_train_params)
        idxh_target = int(idx % self.num_recon)
        input, X_target = self.get_data(idxv, idxh_train, idxh_target)
        return input, X_target
    
    # Label Scaler Download & Reverse Transformation
    def label_unscale(
        self,
        y: np.array or pd.DataFrame
    ):

        # Label Scaler Download & Reverse Usage
        try: self.scaler
        except AttributeError:
            scaler = torch.load(f"{self.settings.datasave_folderpath}/1D Label Scaler (V{self.settings.data_version}).pkl")
        return scaler.inverse_transform(y.reshape(1, -1))

    # --------------------------------------------------------------------------------------------

    # End of Epoch Shuffling Functionality
    def on_epoch_end(self):
        
        # Batch Shuffling
        self.idxv_set = np.arange(len(self.idxv))
        if self.settings.sample_shuffle: np.random.shuffle(self.idxv_set)

        # Reconstruction Parameter Shuffling
        if self.settings.param_shuffle:
            self.idxh_recon = np.hstack((   self.idxh_train[np.sort(np.random.choice(self.num_train_params,
                                            int((self.param_recon * self.num_train_params) / 100), replace = False))],
                                            self.idxh_val[np.sort(np.random.choice(self.num_val_params,
                                            int((self.param_recon * self.num_val_params) / 100), replace = False))]))

    # Batch Data Generation Functionality
    def get_data(self, idxv, idxh_train, idxh_target):

        # Data Access
        data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        X_train = data[idxv, :][:, self.idxh_train[idxh_train]].reshape((len(idxv), 1))                             # [batch_size,  1] Training Data
        y_train = self.params.iloc[self.idxh_train[idxh_train]].values.reshape((1, self.settings.num_labels))       # [1,           num_labels] Training Parameters
        y_target = self.params.iloc[self.idxh_recon[idxh_target]].values.reshape((1, self.settings.num_labels))     # [1,           num_labels] Target Parameters
        y_train = y_train.repeat(len(idxv), 0); y_target = y_target.repeat(len(idxv), 0);                           # [batch_size,  num_labels] Training & Target Parameters
        X_target = data[idxv, :][:, self.idxh_recon[idxh_target]].reshape((len(idxv), 1))                           # [batch_size,  1] GT Target Data
        input = tf.keras.layers.concatenate([X_train, y_train, y_target], axis = 1)                                 # [batch_size,  1 + (2 * num_labels)] Input
        return input, X_target
    
    # Patient Data Generation Functionality
    def get_patient(settings, num_patient: int):

        # Patient Data Access
        data = h5py.File(settings.data_filepath, 'r').get('data1')
        idxv = pd.read_csv( settings.info_filepath,             # List of Index Values ...
                            index_col = 0).to_numpy()           # ... pertaining to each Patient
        idxv = idxv[np.isin(idxv[:, 1], num_patient), 0]        # Patient-Specific Index Values
        X = data[idxv, :]

        # Patient Mask Access
        mask_filepath = Path(f"{settings.mask_folderpath}/p{num_patient}.nii")
        assert(mask_filepath.exists()), f"ERROR: Patient {num_patient}'s Mask not Found!"
        mask = load_img(mask_filepath)
        img = unmask(X.T, mask).get_fdata().T
        return X, mask, img
