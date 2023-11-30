# Library Imports
import os
import pickle
import random
import numpy as np
import argparse
import pandas as pd
import torch
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
class MUDI_1D(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        #info_filepath,         # header_.csv
        settings: argparse.ArgumentParser
    ):
        
        # Parameter & Patient Index Value Access
        super(MUDI_1D).__init__()
        self.settings = settings; self.version = self.settings.version
        self.idx = pd.read_csv( self.settings.info_filepath,            # List of Index Values ...
                                index_col = 0).to_numpy()               # ... pertaining to each Patient
        self.params = pd.read_excel(self.settings.param_filepath)       # List of Dataset's Parameters
        self.num_params = self.params.shape[0]                          # Total Number of Parameters in Dataset
        self.num_patients = len(self.settings.patient_list)             # Total Number of Patients in Dataset
        self.num_train_params = self.settings.num_train_params          # Number of Training Parameters
        self.num_val_params = self.num_params - self.num_train_params   # Number of Validation Parameters

        # ----------------------------------------------------------------------------------------
        
        # Dataset Sample & Label Handling Settings
        self.num_labels = self.settings.num_labels
        if self.settings.gradient_coord:
            self.params = self.params.drop(columns = ['Gradient theta', 'Gradient phi'])                # Choosing of Gradient Orientation
        else: self.params = self.params.drop(columns = ['Gradient x', 'Gradient y', 'Gradient z'])      # from 3D Cartesian or Polar Referential
        assert(self.params.shape[1] == self.num_labels), "ERROR: Labels wrongly Deleted!"
        if self.settings.label_norm:                                                                    # Control Boolean Value for the Normalization of Labels
            self.scaler = StandardScaler()
            self.params = pd.DataFrame( self.scaler.fit_transform(self.params),
                                        columns = self.params.columns)
        
        # ----------------------------------------------------------------------------------------

        # Data Splitting
        self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        self.idxv_train, self.idxv_test = self.vsplit()
        self.idxh_train, self.idxh_val = self.hsplit()

    ##############################################################################################
    # -------------------------------------- Data Splitting --------------------------------------
    ##############################################################################################

    # Vertical / Patient Data Split
    def vsplit(self):

        # Test Patient(s) Selection Feature
        self.num_test_patients = self.settings.num_test_patients                # Number of Test Set Patients
        self.num_train_patients = self.num_patients - self.num_test_patients    # Number of Training Set Patients
        patient_list = self.settings.patient_list.copy(); patient_list.sort()   # Complete Sorted Patient List
        
        # [Option #1] Selected Test Patient(s)
        if self.settings.sel_test_patients is not None:
            self.test_patients = self.settings.sel_test_patients
            self.train_patients = patient_list.copy()
            #for i in self.test_patients: self.train_patients.remove(i)
            self.train_patients.remove(self.test_patients)
        
        # [Option #2] Non-Specific Test Patient(s)
        else:
            if self.settings.patient_shuffle: random.shuffle(patient_list)      # Patient Shuffling Feature
            self.train_patients = patient_list[0 : self.num_train_patients]
            self.test_patients = patient_list[self.num_train_patients : :]
            self.train_patients.sort(); self.test_patients.sort()

        # Training & Test Patient Sample Indexes
        idx_train = self.idx[np.isin(self.idx[:, 1], self.train_patients), 0]
        idx_test = self.idx[np.isin(self.idx[:, 1], self.test_patients), 0]
        return idx_train, idx_test

    # --------------------------------------------------------------------------------------------

    # Horizontal / Parameter Data Split
    def hsplit(self):

        # [Option #1] Existing Selection of Set Labels
        self.idxh_train_filepath = Path(f"{self.settings.save_folderpath}/1D Training Labels (V{self.version}).txt")    # Filepath for Selected Training Labels
        self.idxh_val_filepath = Path(f"{self.settings.save_folderpath}/1D Validation Labels (V{self.version}).txt")    # Filepath for Selected Validation Labels
        if self.idxh_train_filepath.exists():
            print(f"DOWNLOADING Training Parameters (Version {self.settings.version})")
            idx_train = np.sort(np.loadtxt(self.idxh_train_filepath)).astype(int)
            if self.idxh_val_filepath.exists():
                idx_val = np.sort(np.loadtxt(self.idxh_val_filepath)).astype(int)
            else: idx_val = np.setdiff1d(   np.array(range(0, self.num_params)),
                                            idx_train).astype(int)
        
        # [Option #2] Non-Existing Selection of Set Labels
        else:
            if self.settings.sample_shuffle:                        # Sample Shuffling Feature
                idx_y = np.random.permutation(self.num_params)
            else: idx_y = np.array(range(0, self.num_params))
            idx_train = idx_y[0 : self.num_train_params]
            idx_val = idx_y[self.num_train_params : self.num_params]
        return idx_train, idx_val

    ##############################################################################################
    # --------------------------------- Label & Feature Handling ---------------------------------
    ##############################################################################################

    # Label Scaler Download & Reverse Transformation
    def label_unscale(
        path: Path,
        version: int,
        y: np.array or pd.DataFrame
    ):

        # Label Scaler Download & Reverse Usage
        try: scaler
        except NameError:
            scaler = torch.load(f"{path}/1D Label Scaler (V{version}).pkl")
        return scaler.inverse_transform(y.reshape(1, -1))
    
    # --------------------------------------------------------------------------------------------

    # Dataset Feature Indexing Functionality
    def get_data(
        self,
        idxv: int or np.array,              # Vertical Selected Indexes
        idxh: int or np.array = None        # Horizontal Selected Indexes
    ): 
        # Original Dataset File Download
        try: self.data
        except AttributeError:
            self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        
        # Presence of Horizontal Index Selection
        if idxh is not None: data = torch.Tensor(self.data[idxv, :])[:, idxh]
        else: data = torch.Tensor(self.data[idxv, :])

        # Horizontal Mode Patient Splitting
        #for p in self.train_patients:
            #data = 

        # Horizontal / Vertical Mode Setup
        if self.settings.mode == 'h': return data.T
        elif self.settings.mode == 'v': return data.ravel()
        else: return data

    # Dataset Label / Parameter Indexing Functionality
    def get_params(
        self,
        idxv: int or np.array = None,       # Vertical Selected Indexes
        idxh: int or np.array = None        # Horizontal Selected Indexes
    ):

        # Label / Parameter Indexing
        y = self.params.iloc[idxh, :]
        y = torch.Tensor(np.array(y))
        if self.settings.mode == 'v':
            y = y.repeat(len(idxv), 1)
        return y

    # Patient Data Acquisition Functionality
    def get_patient(
        self,
        num_patient: int,
    ):
        
        # Original Dataset File Download
        try: self.data
        except AttributeError:
            self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')

        # Full Patient Data Acquisition
        assert(num_patient in self.settings.patient_list
        ), f"ERROR: Chosen Patient {num_patient} is not contained in the Dataset provided!"
        idxv = self.idx[np.isin(self.idx[:, 1], num_patient), 0]
        pX = self.get_data(idxv)    #torch.Tensor(self.data[idxv, :])

        # Patient Mask Acquisition
        mask_filepath = Path(f"{self.settings.mask_folderpath}/p{num_patient}.nii")
        assert(mask_filepath.exists()
        ), f"ERROR: Mask for Patient {num_patient} is not contained in the Dataset provided!"
        pMask = load_img(mask_filepath)
        #X = unmask(pX.T, pMask); X = X.get_fdata() 
        return pX, pMask

    # --------------------------------------------------------------------------------------------

    # DataLoader Construction Functionality
    def get_loader(
        self,
        idxv: int or np.array,              # Vertical Selected Indexes
        idxh: int or np.array = None        # Horizontal Selected Indexes
    ):  
        # Original Dataset File Download
        try: self.data
        except AttributeError:
            self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')

        # Horizontal Mode DataLoader Construction
        if self.settings.mode == 'h':
            return      DataLoader(
                            TensorDataset(  self.get_data(idxv, self.idxh_train).T,
                                            self.get_data(idxv, self.idxh_val).T), shuffle = False,
                                            num_workers = self.settings.num_workers,
                                            batch_size = self.settings.batch_size)
        
        # Vertical Mode DataLoader Construction
        elif self.settings.mode == 'v':
            return      DataLoader(
                            TensorDataset(  self.get_data(idxv, idxh),
                                            self.get_params(idxv, idxh)),
                                            num_workers = self.settings.num_workers,
                                            batch_size = self.settings.batch_size
                                            * len(idxh), shuffle = False)
        
        # Horizontal/Verti
        # cal  Mode DataLoader Construction
        else: return    DataLoader(
                            TensorDataset(  self.get_data(idxv, self.idxh_train),
                                            self.get_data(idxv, self.idxh_val)), shuffle = False,
                                            num_workers = self.settings.num_workers,
                                            batch_size = self.settings.batch_size)

    ##############################################################################################
    # ------------------------------------- Saving & Loading -------------------------------------
    ##############################################################################################

    # Dataset Saving Functionality
    def save(self):

        # Dataset Loader Saving (Horizontal / Vertical)
        set_list = {'Train': self.idxv_train, 'Test': self.idxv_test}
        if self.settings.mode == 'v': mode_list = {'Train': self.idxh_train, 'Val': self.idxh_val}
        else: mode_list = {'': None}
        for s in set_list:
            for m in mode_list: torch.save( self.get_loader(set_list[s], mode_list[m]),
                                f"{self.settings.save_folderpath}/1D {s}{m}Loader (V{self.version}).pkl")

        # Index & Label Scaler Saving
        torch.save(self.scaler, f"{self.settings.save_folderpath}/1D Label Scaler (V{self.version}).pkl")
        if not self.idxh_train_filepath.exists(): np.savetxt(self.idxh_train_filepath, np.array(self.idxh_train))
        if not self.idxh_val_filepath.exists(): np.savetxt(self.idxh_val_filepath, np.array(self.idxh_val))

        # Full Dataset Class Saving
        f = open(f'{self.settings.save_folderpath}/Horizontal 1D MUDI (Version {self.version})', 'wb')
        del self.data; pickle.dump(self, f); f.close

    # ----------------------------------------------------------------------------------------------------------------------------

    # Dataset Loading Function
    def load(
        path: Path,
        version: int = 0,
    ):
        f = open(Path(f'{path}/Horizontal 1D MUDI (Version {version})'), 'rb')
        mudi = pickle.load(f); f.close
        return mudi
    
    def load_(
        path: Path,
        version: int = 0,
    ):
        filepath = PureWindowsPath(f'{path}/Horizontal 1D MUDI (Version {version}).pth')
        with open(Path(filepath), 'rb') as f:
            mudi = pickle.load(f)
        return mudi

    # Dataset Loader Loading Function
    def loader(
        path: Path,
        version: int = 0,
        set_: str = 'Train',
        mode_: str = ''
    ):  return torch.load(f"{path}/1D {set_}{mode_}Loader (V{version}).pkl")
