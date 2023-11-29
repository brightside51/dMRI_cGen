# Library Imports
import random
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

# 2D MUDI Dataset Initialization Class (Random)
class MUDI_CVAE_GAN_2D(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        subject: int or list,
        random: bool = True,
        mode: str = 'Target',
        source_param: int or float = 100,
        target_param: int or float = 100,
        target_slice: int or float = 100,
        param_loop: int or floar = 100
    ):
        
        # Parameter & Patient Index Value Access
        super(MUDI_CVAE_GAN_2D).__init__(); self.settings = settings
        self.random = random; self.mode = mode; self.source_param = source_param
        self.target_param = target_param; self.target_slice = target_slice
        self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        self.params = pd.read_excel(self.settings.param_filepath)

        # Vertical Splitting (Patient & Slice Selection)
        self.idxv = pd.read_csv(self.settings.info_filepath,                    # List of Index Values ...
                                index_col = 0).to_numpy()                       # ... pertaining to each Patient
        self.idxv = self.idxv[np.isin(self.idxv[:, 1], subject), 0]             # Patient-Specific Index Values
        self.mask = MUDI_CVAE_GAN_2D.get_mask(self.settings, subject[0])        # Patient-Specific Mask Download
        self.data = unmask(self.data[self.idxv, :].T, self.mask).get_fdata().T  # Patient-Specific Image Unmasking
        self.data = MUDI_CVAE_GAN_2D.zero_padding(self.data, self.settings.img_shape)   # Image Pre-Processing: Zero Padding
        self.idxv_slice_full = np.arange(self.mask.shape[-1])                   # Full List of Available Image Slices

        # Horizontal Splitting (Source & Target Parameter Selection)
        idxh_source_filepath = Path(f"{self.settings.datasave_folderpath}/Training Labels (V{self.settings.data_version}).txt")
        idxh_target_filepath = Path(f"{self.settings.datasave_folderpath}/{self.mode} Labels (V{self.settings.data_version}).txt")
        self.idxh_source_full = np.sort(np.loadtxt(idxh_source_filepath)).astype(int)
        if not idxh_target_filepath.exists(): self.idxh_target_full = self.label_gen()
        else: self.idxh_target_full = np.sort(np.loadtxt(idxh_target_filepath)).astype(int)

        # Vertical & Horizontal Sub-Sectioning & Shuffling
        self.s_target = int((self.target_slice * len(self.idxv_slice_full)) / 100)
        print(f"     > Utilizing {self.s_target} \ {len(self.idxv_slice_full)} of the Available Slices")
        self.h_source = int((self.source_param * len(self.idxh_source_full)) / 100)
        self.h_target = int((self.target_param * len(self.idxh_target_full)) / 100)
        self.h_target = int((self.target_param * len(self.idxh_target_full)) / 100)
        self.num_combo = int((param_loop * (self.h_source * self.h_target)) / 100); self.shuffle(init = True)
        print(f"     > Utilizing {self.h_source} \ {len(self.idxh_source_full)} of the Training Parameters")
        print(f"     > Utilizing {self.h_target} \ {len(self.idxh_target_full)} of the Target Parameters")
        print(f"     > Pre-Processing Images to be of Square Shape of {self.settings.img_shape}")
        print(f"     > Looping through {self.num_combo} \ {self.h_source * self.h_target} Source / Target Parameter Combos")

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
    def __len__(self): return self.s_target * self.num_combo

    # Image Zero-Padding Pre-Processing Method Functionality
    def zero_padding(
        data: np.array or torch.Tensor,
        img_shape: int
    ):

        # Input Data Assertions
        assert(data.ndim >= 3), "ERROR: Pre-Processing Input Data has the Wrong Dimmensions"
        assert(img_shape >= int(data.shape[-2]) and img_shape >= int(data.shape[-1])
        ), "ERROR: Pre-Processed Output Data Shape is Larget than Input one!"

        # Zero-Padding Implementation
        padding = np.array([0, 0, (img_shape - data.shape[-2]) / 2, (img_shape - data.shape[-1]) / 2])
        padding = padding.reshape((1, -1)).T + np.array([0, 0])
        padding[:, -1] = np.ceil(padding[:, -1]); padding[:, -2] = np.floor(padding[:, -2])
        return np.pad(data, padding.astype(np.int32), 'constant')
        
    # Single-Batch Generation Functionality
    def __getitem__(self, idx) -> tuple[np.ndarray, np.float32]:
        
        # Batch Vertical/Slice & Horizontal/Parameter Indexing
        idxv_slice = idx % self.s_target                                # Batch's Vertical Index for X_train
        idxh_loop = (idx // self.s_target) % self.num_combo             # Batch's Horizontal Index for Loop Combo
        if self.random:
            idxh_source = random.randrange(self.h_source)               # Random Batch's Horizontal Index for y_train
            idxh_target = random.randrange(self.h_target)               # Random Batch's Horizontal Index for y_target
        else:
            idxh_source = int(np.where(self.idxh_source == self.idxh_combo[idxh_loop][0])[0])   # Fixed y_train Batch's Horizontal Index
            idxh_target = int(np.where(self.idxh_target == self.idxh_combo[idxh_loop][1])[0])   # Fixed y_target Batch's Horizontal Index
        #print( f"Item #{idx} | Slice #{idxv_slice} | Source Parameter #{self.idxh_source[idxh_source]}" +\
        #        "| Target Parameter #{self.idxh_target[idxh_target]}")

        # Batch Data Generation
        X_train = self.data[self.idxh_source[idxh_source], self.idxv_slice[idxv_slice], :, :]  # [1, 1, :, :] Training Image Slice Data
        y_train = self.params.iloc[self.idxh_source[idxh_source]].values                       # [num_labels] Training Image Parameter Values
        X_target = self.data[self.idxh_target[idxh_target], self.idxv_slice[idxv_slice], :, :] # [1, 1, :, :] GT Target Image Slice Data
        y_target = self.params.iloc[self.idxh_target[idxh_target]].values                      # [num_labels] Target Image Parameter Values
        return {'X_train': np.array(X_train).reshape((1, X_train.shape[0], X_train.shape[1])).astype(np.float32),
                'X_target': np.array(X_target).reshape((1, X_target.shape[0], X_target.shape[1])).astype(np.float32),
                'y_train': np.ravel(y_train).astype(np.float32), 'y_target': np.ravel(y_target).astype(np.float32),
                'idxv_slice': idxv_slice, 'slice_target': self.idxv_slice[idxv_slice], 'idxh_loop': idxh_loop,
                'idxh_source': idxh_source, 'param_source': self.idxh_source[idxh_source],
                'idxh_target': idxh_target, 'param_target': self.idxh_target[idxh_target]}
    
    # --------------------------------------------------------------------------------------------

    # Source Parameter Shuffling Functionality (tbu at Epoch's Beggining)
    def source_param_shuffle(
        self,
        init: bool = False,
        idxh_source: np.array = None
    ):
        if idxh_source is not None: self.idxh_source = idxh_source
        else:
            if self.settings.param_shuffle or init:
                self.idxh_source = self.idxh_source_full[   np.sort(np.random.choice(len(self.idxh_source_full),
                                                            self.h_source, replace = False))]

    # Target Parameter Shuffling Functionality (tbu at Epoch's Beggining)
    def target_param_shuffle(
        self,
        init: bool = False,
        idxh_target: np.array = None
    ):
        if idxh_target is not None: self.idxh_target = idxh_target
        else:
            if self.settings.param_shuffle or init:
                self.idxh_target = self.idxh_target_full[   np.sort(np.random.choice(len(self.idxh_target_full),
                                                            self.h_target, replace = False))]
                
    # Target Slice Shuffling Functionality (tbu at Epoch's Beggining)
    def slice_shuffle(
        self,
        init: bool = False,
        idxv_slice: np.array = None
    ):  
        if idxv_slice is not None: self.idxv_slice = idxv_slice
        else:
            if self.settings.slice_shuffle or init:
                self.idxv_slice = self.idxv_slice_full[     np.sort(np.random.choice(len(self.idxv_slice_full),
                                                            self.s_target, replace = False))]

    # Target Parameter Shuffling Functionality (tbu at Epoch's Beggining)
    def combo_shuffle(
        self,
        init: bool = False,
        idxh_combo: np.array = None
    ):
        if idxh_combo is not None: self.idxh_combo = idxh_combo
        else:
            if self.settings.param_shuffle or init:
                self.idxh_combo = []
                for i in range(self.num_combo):
                    self.idxh_combo.append(np.array([self.idxh_source[random.randrange(self.h_source)],
                                                    self.idxh_target[random.randrange(self.h_target)]]))
    
    # Target Voxel & Parameter Shuffling Functionality (tbu at Epoch's Beggining)
    def shuffle(
        self,
        init: bool = False,
        idxh_source: np.array = None,
        idxh_target: np.array = None,
        idxv_slice: np.array = None,
        idxh_combo: np.array = None
    ):
        self.source_param_shuffle(init, idxh_source)
        self.target_param_shuffle(init, idxh_target)
        self.slice_shuffle(init, idxv_slice)
        self.combo_shuffle(init, idxh_combo)

    # --------------------------------------------------------------------------------------------

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
            print(f">     Saving File Target Parameters for Training ({len(idxh_target)}) & Test {len(idxh_test)} Set's")
            np.savetxt(Path(f"{self.settings.datasave_folderpath}/Test Labels (V{self.settings.data_version}).txt"), idxh_test)
            np.savetxt(Path(f"{self.settings.datasave_folderpath}/Target Labels (V{self.settings.data_version}).txt"), idxh_target)
        if self.mode == 'Target': return idxh_target
        else: return idxh_test
        
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

    # Cartesian to Polar Coordinate Conversion Functionality
    def cart2polar(self, y: np.array):

        # Cartesian to Polar Coordinates Conversion
        phi = np.arccos(y[:, 2]).astype(np.float32); theta = np.arctan2(y[:, 1], y[:, 0])
        theta = np.where(theta < 0, 2 * np.pi - np.abs(theta), theta).astype(np.float32)
        theta = np.expand_dims(theta, axis = 1); phi = np.expand_dims(phi, axis = 1)
        return pd.DataFrame(data = np.concatenate((theta, phi), axis = 1),
                            columns = ['Gradient theta', 'Gradient phi'])

    # Patient Mask Retrieval Functionality
    def get_mask(settings, patient_id: int):
        mask_filepath = Path(f"{settings.mask_folderpath}/p{patient_id}.nii")
        assert(mask_filepath.exists()), f"ERROR: Patient {patient_id}'s Mask not Found!"
        return load_img(mask_filepath)
    
    # Data Retrieval Functionality
    def get_data(
        settings,
        patient_id: int,
        idxv: int or np.array = None,
        idxh_source: int or np.array = None,
        idxh_target: int or np.array = None
    ):
        
        # Dataset Download & Patient-Wise Splitting
        idxv = np.expand_dims(np.array(idxv), axis = 1)
        data = h5py.File(settings.data_filepath, 'r').get('data1')
        idxv_full = pd.read_csv(settings.info_filepath, index_col = 0).to_numpy()
        idxv_full = idxv_full[np.isin(idxv_full[:, 1], patient_id), 0]                  # Patient-Specific Index Values
        mask = MUDI_CVAE_GAN_2D.get_mask(settings, patient_id)                          # Patient-Specific Mask Download
        data = unmask(data[idxv_full, :].T, mask).get_fdata().T                         # Patient-Specific Data Unmasking
        
        # Dataset Source Horizontal / Parameter Indexing
        if idxh_source is not None:
            idxh_source = np.expand_dims(np.array(idxh_source), axis = 1)
            X_source = data[tuple(np.concatenate([idxh_source, idxv], axis = -1).T)]
            X_source = np.expand_dims(X_source, axis = 1)
            X_source = MUDI_CVAE_GAN_2D.zero_padding(X_source, settings.img_shape)            
        else: X_source = None

        # Dataset Target Horizontal / Parameter & Vertical / Slice Indexing
        if idxh_target is not None:
            idxh_target = np.expand_dims(np.array(idxh_target), axis = 1)
            X_target = data[tuple(np.concatenate([idxh_target, idxv], axis = -1).T)]
            X_target = np.expand_dims(X_target, axis = 1)
            X_target = MUDI_CVAE_GAN_2D.zero_padding(X_target, settings.img_shape)   
        else: X_target = None
        return torch.Tensor(X_source), torch.Tensor(X_target)

    