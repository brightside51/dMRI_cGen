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
from pathlib import Path, PureWindowsPath
from torch.utils.data import DataLoader, Dataset, TensorDataset
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

# Horizontal 1D MUDI Dataset Initialization Class
class MRISelectorSubjDataset(keras.utils.Sequence):
    """MRI dataset to select features from."""

    def __init__(self, root_dir, dataf, headerf, subj_list, batch_size=100, shuffle=False):
        """
        Args:
            root_dir (string): Directory with the .csv files
            data (string): Data .csv file
            header (string): Header .csv file
            subj_list (list): list of all the subjects to include
        """
        
        self.root_dir = root_dir
        self.dataf = dataf
        self.headerf = headerf
        self.subj_list = subj_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # load the header
        subj = self.subj_list[0]
        self.header = pd.read_csv(os.path.join(self.root_dir,
                                             self.headerf), index_col=0).to_numpy()
        self.ind = self.header[np.isin(self.header[:,1],self.subj_list),0]
#         print(self.ind)
        
        self.indexes = np.arange(len(self.ind)) 

    def __len__(self):
        return int(np.ceil(len(self.ind) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         print(indexes)

        # Find list of IDs
        list_IDs_temp = [self.ind[k] for k in indexes]
#         print(list_IDs_temp)

        # Generate data
        signals = self.__data_generation(list_IDs_temp)

        return signals, signals

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ind)) 
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        # X = pd.read_csv(os.path.join(self.root_dir, self.dataf), index_col=0, skiprows=lambda x: x not in list_IDs_temp).to_numpy()
        h5f = h5py.File(os.path.join(self.root_dir, self.dataf), 'r')
        X = h5f.get('data1')
        X = X[list_IDs_temp,:]

        return X

class MRIDecoderSubjDataset(keras.utils.Sequence):
    """MRI dataset to select features from."""

    def __init__(self, root_dir, dataf, headerf, subj_list, selecf, input_dim=1344, batch_size=100, shuffle=False):
        """
        Args:
            root_dir (string): Directory with the .csv files
            data (string): Data .csv file
            header (string): Header .csv file
            subj_list (list): list of all the subjects to include
        """
        
        self.root_dir = root_dir
        self.dataf = dataf
        self.headerf = headerf
        self.subj_list = subj_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.selecf = selecf
        
        # load the header
        subj = self.subj_list[0]
        self.header = pd.read_csv(os.path.join(self.root_dir,
                                             self.headerf), index_col=0).to_numpy()
        self.ind = self.header[np.isin(self.header[:,1],self.subj_list),0]
#         print(self.ind)
        
        self.indexes = np.arange(len(self.ind)) 
        
        # load the select file
        self.selecind = np.sort(np.loadtxt(self.selecf)).astype(int)
        #print(self.selecind)
        
        self.msk = np.logical_not(np.isin(np.arange(input_dim),self.selecind))

    def __len__(self):
        return int(np.ceil(len(self.ind) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         print(indexes)

        # Find list of IDs
        list_IDs_temp = [self.ind[k] for k in indexes]
#        print(list_IDs_temp)

        # Generate data
        selec_signals, signals = self.__data_generation(list_IDs_temp)
        
#        print(selec_signals.shape)
#        print(signals.shape)

        return selec_signals, signals

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ind)) 
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        # X = pd.read_csv(os.path.join(self.root_dir, self.dataf), index_col=0, skiprows=lambda x: x not in list_IDs_temp).to_numpy()
        h5f = h5py.File(os.path.join(self.root_dir, self.dataf), 'r')
        X = h5f.get('data1')
        X = X[list_IDs_temp,:]
        X1 = X[:,self.selecind]
        # X2 = X[:,self.msk]
        return X1, X
