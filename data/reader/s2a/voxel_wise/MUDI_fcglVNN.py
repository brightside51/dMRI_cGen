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

# 1D MUDI Dataset Initialization Class (V1)
class MUDI_fcglVNN(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        subject: int or list,
        mode: str = 'Train',
        target_param: int or float = 100,
        target_voxel: int or float = 100
    ):
        
        # Parameter & Patient Index Value Access
        super(MUDI_fcglVNN).__init__()
        self.settings = settings; self.mode = mode
        if mode == 'Train': self.mode = 'Target'
        self.target_param = target_param; self.target_voxel = target_voxel
        self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        self.params = pd.read_excel(self.settings.param_filepath)

        # Vertical Splitting (Patient Selection)
        self.idxv = pd.read_csv(self.settings.info_filepath,            # List of Index Values ...
                                index_col = 0).to_numpy()               # ... pertaining to each Patient
        self.idxv = self.idxv[np.isin(self.idxv[:, 1], subject), 0]     # Patient-Specific Index Values

        # Horizontal Splitting (Parameter Selection)
        idxh_train_filepath = Path(f"{self.settings.datasave_folderpath}/Training Labels (V{self.settings.data_version}).txt")
        idxh_target_filepath = Path(f"{self.settings.datasave_folderpath}/{self.mode} Labels (V{self.settings.data_version}).txt")
        self.idxh_train = np.sort(np.loadtxt(idxh_train_filepath)).astype(int)
        if not idxh_target_filepath.exists(): self.idxh_target_full = self.label_gen()
        else: self.idxh_target_full = np.sort(np.loadtxt(idxh_target_filepath)).astype(int)

        # Vertical & Horizontal Sub-Sectioning & Shuffling
        self.h_target = int((self.target_param * len(self.idxh_target_full)) / 100)
        self.v_target = int((self.target_voxel * len(self.idxv)) / 100); self.shuffle()
        print(f"     > Utilizing {self.h_target} \ {len(self.idxh_target_full)} of the Target Parameters")
        print(f"     > Utilizing {self.v_target} \ {len(self.idxv)} of the Training Voxels")

        # Parameter Selection & Value Normalization / Scaling
        if self.settings.gradient_coord:
            self.params = self.params.drop(columns = ['Gradient theta', 'Gradient phi'])                # Choosing of Gradient Orientation
        else: self.params = self.params.drop(columns = ['Gradient x', 'Gradient y', 'Gradient z'])
        if self.settings.label_norm == 'auto':
            print(f"     > Automatic Normalization of all {self.settings.num_labels} Parameter Values")
            self.scaler = StandardScaler()
            self.params = pd.DataFrame( self.scaler.fit_transform(self.params),
                                        columns = self.params.columns)
            scaler_filepath = Path(f"{self.settings.datasave_folderpath}/1D Label Scaler (V{self.settings.data_version}).pkl")
            if not scaler_filepath.exists(): torch.save(self.scaler, scaler_filepath)
        elif self.settings.label_norm == 'manual':
            print(f"     > Manual Normalization of all {self.settings.num_labels} Parameter Values")
            self.params[['Gradient theta', 'Gradient phi']] = self.cart2polar(self.params[['Gradient x', 'Gradient y', 'Gradient z']].values)
            self.params = self.params.drop(columns = ['Gradient x', 'Gradient y', 'Gradient z'])
            self.params['b Value'] = self.params['b Value'] / 10000
            self.params['TI'] = self.params['TI'] / 10000
            self.params['TE'] = (self.params['TE'] - 80) / 200
        else: print(f"     > No Normalization of all {self.settings.num_labels} Parameter Values")
        assert(self.params.shape[1] == self.settings.num_labels), "ERROR: Labels wrongly Deleted!"
    
    # --------------------------------------------------------------------------------------------

    # DataLoader Length / No. Batches Computation Functionality
    def __len__(self): return self.v_target * self.h_target

    # Single-Batch Generation Functionality
    def __getitem__(self, idx) -> tuple[np.ndarray, np.float32]:
        
        # [Parameter Batch] Batch Vertical/Patient & Horizontal/Parameter Indexing
        #idxv = idx // self.h_target         # Batch's Vertical Index for X_train
        #idxh = idx % self.h_target          # Batch's Horizontal Index for y_target

        # [Image Batch] Batch Vertical/Patient & Horizontal/Parameter Indexing
        idxv = idx % self.v_target          # Batch's Vertical Index for X_train
        idxh = idx // self.v_target         # Batch's Horizontal Index for y_target
        
        # Batch Data Generation
        X_train = self.data[self.idxv_target[idxv], :][self.idxh_train]             # [num_train_params] Training Data
        y_target = self.params.iloc[self.idxh_target[idxh]].values                  # [num_labels] Target Parameters
        X_target = self.data[self.idxv_target[idxv], :][self.idxh_target[idxh]]     # [    1    ] GT Target Data
        input = np.hstack((X_train, y_target)).astype(np.float32)                   # [num_train_params + num_labels] Input
        return input, X_target
    
    # --------------------------------------------------------------------------------------------

    # Target Parameter Shuffling Functionality (tbu at Epoch's Beggining)
    def param_shuffle(
        self,
        idxh_target: np.array = None
    ):
        if idxh_target is not None: self.idxh_target = idxh_target
        else:
            if self.settings.param_shuffle:
                self.idxh_target = self.idxh_target_full[   np.sort(np.random.choice(len(self.idxh_target_full),
                                                            self.h_target, replace = False))]
                
    # Target Voxel Shuffling Functionality (tbu at Epoch's Beggining)
    def voxel_shuffle(
        self,
        idxv_target: np.array = None
    ):  
        if idxv_target is not None: self.idxv_target = idxv_target
        else:
            if self.settings.voxel_shuffle:
                self.idxv_target = self.idxv[   np.sort(np.random.choice(len(self.idxv),
                                                self.v_target, replace = False))]
    
    # Target Voxel & Parameter Shuffling Functionality (tbu at Epoch's Beggining)
    def shuffle(
        self,
        idxh_target: np.array = None,
        idxv_target: np.array = None
    ):  self.param_shuffle(idxh_target); self.voxel_shuffle(idxv_target)

    # Random Target Parameter for Training & Test Sets Generation Functionality
    def label_gen(self):

        # Target & Test Label Index File Generation
        idxh_target = np.delete(np.arange(self.params.shape[0]), self.idxh_train)
        if self.settings.test_target_param != 0:
            idxh_test = idxh_target[np.sort(np.random.choice(len(idxh_target),
                                    self.settings.test_target_param, replace = False))]
            idxh_target = np.delete(idxh_target, np.where(np.in1d(idxh_target, idxh_test)))
            for i in range(self.settings.test_target_param): assert(idxh_test[i] not in idxh_target
                ), f"ERROR: Target Parameter #{i} for Training & Test Sets not mutually Exclusive"
            
            # Target & Test Label Index File Saving
            print(f">     Saving File Target Parameters for Test {len(idxh_test)} Set")
            np.savetxt(Path(f"{self.settings.datasave_folderpath}/Test Labels (V{self.settings.data_version}).txt"), idxh_test)
        print(f">     Saving File Target Parameters for Target ({len(idxh_target)}) Set")
        np.savetxt(Path(f"{self.settings.datasave_folderpath}/Target Labels (V{self.settings.data_version}).txt"), idxh_target)
        if self.mode == 'Train': return idxh_target
        else: return idxh_test

    # --------------------------------------------------------------------------------------------
        
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
    def get_img(settings, num_patient: int):

        # Patient Data Access
        data = h5py.File(settings.data_filepath, 'r').get('data1')
        idxv = pd.read_csv( settings.info_filepath,             # List of Index Values ...
                            index_col = 0).to_numpy()           # ... pertaining to each Patient
        idxv = idxv[np.isin(idxv[:, 1], num_patient), 0]        # Patient-Specific Index Values
        X = torch.Tensor(data[idxv, :])

        # Patient Mask Access
        mask = MUDI_fcglVNN.get_mask(settings, num_patient = num_patient)
        img = unmask(X.T, mask).get_fdata().T
        return torch.Tensor(img)

    # Patient Mask Retrieval Functionality
    def get_mask(settings, num_patient: int):
        mask_filepath = Path(f"{settings.mask_folderpath}/p{num_patient}.nii")
        assert(mask_filepath.exists()), f"ERROR: Patient {num_patient}'s Mask not Found!"
        #return torch.Tensor(np.array(load_img(mask_filepath).dataobj, dtype = np.float32))
        return load_img(mask_filepath)
    