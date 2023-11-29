# Library Imports
import numpy as np
import argparse
import pandas as pd
import torch
import h5py

# Functionality Import
from pathlib import Path
from nilearn.image import load_img
from nilearn.masking import unmask
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# 1D MUDI Dataset Initialization Class (V0)
class MUDI_fcglVNN(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        subject: int or list,
        param_recon: int or float
    ):
        
        # Parameter & Patient Index Value Access
        super(MUDI_fcglVNN).__init__()
        self.settings = settings; self.param_recon = param_recon
        self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')

        # Vertical Splitting (Patient Selection)
        self.idxv = pd.read_csv(self.settings.info_filepath,            # List of Index Values ...
                                index_col = 0).to_numpy()               # ... pertaining to each Patient
        self.idxv = self.idxv[np.isin(self.idxv[:, 1], subject), 0]     # Patient-Specific Index Values

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
        #assert(self.params.shape[1] == self.num_labels), "ERROR: Labels wrongly Deleted!"

        # Label Normalization / Scaling
        if self.settings.label_norm == 'auto':                                                          # Control Boolean Value for the Normalization of Labels
            print("Automatic Normalization of Parameter Values")
            self.scaler = StandardScaler()
            self.params = pd.DataFrame( self.scaler.fit_transform(self.params),
                                        columns = self.params.columns)
            scaler_filepath = Path(f"{self.settings.datasave_folderpath}/1D Label Scaler (V{self.settings.data_version}).pkl")
            if not scaler_filepath.exists(): torch.save(self.scaler, scaler_filepath)
        
        # Label Manual Normalization
        elif self.settings.label_norm == 'manual':
            print("Manual Normalization of Parameter Values")
            self.params[['Gradient theta', 'Gradient phi']] = self.cart2polar(self.params[['Gradient x', 'Gradient y', 'Gradient z']].values)
            self.params = self.params.drop(columns = ['Gradient x', 'Gradient y', 'Gradient z'])
            self.params['b Value'] = self.params['b Value'] / 10000
            self.params['TI'] = self.params['TI'] / 10000
            self.params['TE'] = (self.params['TE'] - 80) / 200
        else: print("No Normalization of Parameter Values")

        # Random Selection of Parameters for Training Loop Reconstruction
        self.num_train_recon = int((self.param_recon * self.num_train_params) / 100)
        self.num_val_recon = int((self.param_recon * self.num_val_params) / 100)
        self.num_recon = self.num_train_recon + self.num_val_recon
        self.idxh_recon = np.hstack((   self.idxh_train[np.sort(np.random.choice(self.num_train_params,
                                                                self.num_train_recon, replace = False))],
                                        self.idxh_val[np.sort(np.random.choice(self.num_val_params,
                                                                self.num_val_recon, replace = False))]))
    
    # --------------------------------------------------------------------------------------------

    # DataLoader Length / No. Batches Computation Functionality
    def __len__(self): return int(len(self.idxv) * self.num_recon)

    # Single-Batch Generation Functionality
    def __getitem__(self, index) -> tuple[np.ndarray, np.float32]:
        
        # The first 'num_recon' Batches will contain the exact same Training Data 'X_train', but have its 
        # Target Parameter 'y_target' and Target Voxel Intensity GT 'X_target' changed, according to which
        # Parameter the Training Data is being mapped to, allowing for a normal Training Step.

        # Batch Vertical/Patient & Horizontal/Parameter Indexing
        idxv = index % len(self.idxv)       # Batch's Vertical Index for X_train
        idxh = index // len(self.idxv)      # Batch's Horizontal Index for y_target
        
        # Batch Data Generation
        X_train = self.data[self.idxv[idxv], :][self.idxh_train]                # [num_train_params] Training Data
        y_target = self.params.iloc[self.idxh_recon[idxh]].values               # [num_labels] Target Parameters
        X_target = self.data[self.idxv[idxv], :][self.idxh_recon[idxh]]         # [    1    ] GT Target Data
        input = np.hstack((X_train, y_target)).astype(np.float32)               # [num_train_params + num_labels] Input
        return input, X_target
    
    # --------------------------------------------------------------------------------------------

    # End of Epoch Reconstruction Parameter Shuffling Functionality
    def on_epoch_end(
        self,
        idxh_recon: np.array = None
    ):
        
        # Batch Shuffling
        self.idxv_set = np.arange(len(self.idxv))
        if self.settings.sample_shuffle: np.random.shuffle(self.idxv_set)

        # Reconstruction Parameter Shuffling
        if idxh_recon is not None: self.idxh_recon = idxh_recon
        else:
            if self.settings.param_shuffle:
                self.idxh_recon = np.hstack((   self.idxh_train[np.sort(np.random.choice(self.num_train_params,
                                                int((self.param_recon * self.num_train_params) / 100), replace = False))],
                                                self.idxh_val[np.sort(np.random.choice(self.num_val_params,
                                                int((self.param_recon * self.num_val_params) / 100), replace = False))]))
        
    
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

    # Cartesian to Polar Coordinate Conversion Functionality
    def cart2polar(self, y: np.array):

        # Cartesian to Polar Coordinates Conversion
        phi = np.arccos(y[:, 2]).astype(np.float32); theta = np.arctan2(y[:, 1], y[:, 0])
        theta = np.where(theta < 0, 2 * np.pi - np.abs(theta), theta).astype(np.float32)
        theta = np.expand_dims(theta, axis = 1); phi = np.expand_dims(phi, axis = 1)
        return pd.DataFrame(data = np.concatenate((theta, phi), axis = 1),
                            columns = ['Gradient theta', 'Gradient phi'])

    # --------------------------------------------------------------------------------------------
    
    # Patient Data Generation Functionality
    def get_patient(settings, num_patient: int):

        # Patient Data Access
        data = h5py.File(settings.data_filepath, 'r').get('data1')
        idxv = pd.read_csv( settings.info_filepath,             # List of Index Values ...
                            index_col = 0).to_numpy()           # ... pertaining to each Patient
        idxv = idxv[np.isin(idxv[:, 1], num_patient), 0]        # Patient-Specific Index Values
        X = torch.Tensor(data[idxv, :])

        # Patient Mask Access
        mask = MUDI_fcglVNN.get_mask(settings, num_patient = num_patient)
        img = unmask(X.T, mask).get_fdata().T
        return X, mask, torch.Tensor(img)

    # Patient Mask Retrieval Functionality
    def get_mask(settings, num_patient: int):
        mask_filepath = Path(f"{settings.mask_folderpath}/p{num_patient}.nii")
        assert(mask_filepath.exists()), f"ERROR: Patient {num_patient}'s Mask not Found!"
        #return torch.Tensor(np.array(load_img(mask_filepath).dataobj, dtype = np.float32))
        return load_img(mask_filepath)
    