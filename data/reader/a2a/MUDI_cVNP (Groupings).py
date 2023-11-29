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

# cVNP MUDI Dataset Initialization Class (V0)
class MUDI_cVNP(Dataset):

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,
        subject: int or list,
        mode: str = 'Target',
        source_param: int or float = 100,
        target_param: int or float = 100,
        target_voxel: int or float = 100,
        sample_size: int = 0,
    ):
        
        # Parameter & Patient Index Value Access
        super(MUDI_cVNP).__init__()
        self.settings = settings; self.source_param = source_param
        self.target_param = target_param; self.target_voxel = target_voxel
        self.data = h5py.File(self.settings.data_filepath, 'r').get('data1')
        self.params = pd.read_excel(self.settings.param_filepath)

        # Vertical Splitting (Patient Selection)
        self.idxv = pd.read_csv(self.settings.info_filepath,            # List of Index Values ...
                                index_col = 0).to_numpy()               # ... pertaining to each Patient
        self.idxv = self.idxv[np.isin(self.idxv[:, 1], subject), 0]     # Patient-Specific Index Values

        # Horizontal Splitting (Parameter Selection)
        idxh_source_filepath = Path(f"{self.settings.datasave_folderpath}/Training Labels (V{self.settings.data_version}).txt")
        idxh_target_filepath = Path(f"{self.settings.datasave_folderpath}/{mode} Labels (V{self.settings.data_version}).txt")
        self.idxh_source_base = np.sort(np.loadtxt(idxh_source_filepath)).astype(int)
        if not idxh_target_filepath.exists(): self.label_gen()
        self.idxh_target_base = np.sort(np.loadtxt(idxh_target_filepath)).astype(int)

        # Vertical & Horizontal Sub-Sectioning & Shuffling
        self.h_source = int((self.source_param * len(self.idxh_source_base)) / 100)
        print(f"     > Utilizing {self.h_source} \ {len(self.idxh_source_base)} of the Training Parameters")
        self.h_target = int((self.target_param * len(self.idxh_target_base)) / 100)
        print(f"     > Utilizing {self.h_target} \ {len(self.idxh_target_base)} of the Target Parameters")
        self.v_target = int((self.target_voxel * len(self.idxv)) / 100); self.shuffle(init = True)
        
        # Voxel Grouping Functionality (for Loader Step Reduction)
        # https://discuss.pytorch.org/t/data-loader-process-killed/120181
        if sample_size == 0: self.sample_size = self.v_target               # Automatic Sample Grouping Size 
        else: self.sample_size = sample_size
        print(f"     > Utilizing {self.v_target} \ {len(self.idxv)} of the Training Samples in Groupings of {self.sample_size}")
        self.g_target = int(np.ceil(self.v_target / self.sample_size))      # Number of Sample Groupings in a Full Image

        # Parameter Selection & Value Normalization / Scaling
        if self.settings.gradient_coord:
            self.params = self.params.drop(columns = ['Gradient theta', 'Gradient phi'])
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
    def __len__(self): return int(self.g_target * self.h_source * self.h_target)

    # Single-Batch Generation Functionality
    def __getitem__(self, idx) -> tuple[np.ndarray, np.float32]:

        # Each Voxel Intensity Value 'X_train' will have its corresponding Training Parameter
        # 'y_train' and be mapped to one of the 'num_param_to' the Target Parameter 'y_target'.
        # This means that each voxel position will see all the 'num_param_from' Parameters in it
        # be mapped to the 'num_param_to' Parameters in the Validation Set

        # The order goes from a Minor to a Major Scale, meaning it will use all Voxels of the same Image
        # cosecutively, and then select one Target Parameter to map to, and iterate through all Source
        # Parameters, only changing the Target when these all these are exhausted

        # [Destination -> Origin -> Voxel] Batch Vertical/Patient & Horizontal/Parameter Indexing
        idxg = idx % self.g_target                                      # Batch's Sample Grouping Index
        idxh_source = (idx // self.g_target) % self.h_source            # Batch's Horizontal Index for y_train
        idxh_target = idx // (self.g_target * self.h_source)            # Batch's Horizontal Index for y_target
        if idxg == self.g_target - 1: idxv = self.idxv_target[idxg * self.sample_size ::]
        else: idxv = self.idxv_target[idxg * self.sample_size : (idxg + 1) * self.sample_size]

        # Batch Data Generation
        X_train = self.data[idxv, :][:, self.idxh_source[idxh_source]]          # [    1                   ] Training Data
        y_train = self.params.iloc[self.idxh_source[idxh_source]].values        # [num_labels              ] Training Parameters
        y_train = np.repeat(y_train.reshape(1, -1), len(idxv), axis = 0)        # [num_labels * sample_size] Training Parameters
        y_target = self.params.iloc[self.idxh_target[idxh_target]].values       # [num_labels              ] Target Parameters
        y_target = np.repeat(y_target.reshape(1, -1), len(idxv), axis = 0)      # [num_labels * sample_size] Target Parameters
        X_target = self.data[idxv, :][:, self.idxh_target[idxh_target]]         # [    1                   ] GT Target Data
        return {'X_train': np.array(X_train),
                'X_target': np.array(X_target),
                'idxv': idxv, 'idxg': idxg, 'g_size': len(idxv),
                'y_train': np.ravel(y_train).astype(np.float32),
                'param_source': self.idxh_source[idxh_source],
                'idxh_source': idxh_source,
                'y_target': np.ravel(y_target).astype(np.float32),
                'param_target': self.idxh_target[idxh_target],
                'idxh_target': idxh_target}
    
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
                self.idxh_source = self.idxh_source_base[   np.sort(np.random.choice(len(self.idxh_source_base),
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
                self.idxh_target = self.idxh_target_base[   np.sort(np.random.choice(len(self.idxh_target_base),
                                                            self.h_target, replace = False))]
                
    # Target Voxel Shuffling Functionality (tbu at Epoch's Beggining)
    def voxel_shuffle(
        self,
        init: bool = False,
        idxv_target: np.array = None
    ):  
        if idxv_target is not None: self.idxv_target = idxv_target
        else:
            if self.settings.voxel_shuffle or init:
                self.idxv_target = self.idxv[   np.sort(np.random.choice(len(self.idxv),
                                                self.v_target, replace = False))]
    
    # All Selected Voxel & Parameter Shuffling Functionality (tbu at Epoch's Beggining)
    def shuffle(
        self,
        init: bool = False,
        idxh_source: np.array = None,
        idxh_target: np.array = None,
        idxv_target: np.array = None
    ):
        self.source_param_shuffle(init, idxh_source)
        self.target_param_shuffle(init, idxh_target)
        self.voxel_shuffle(init, idxv_target)

    # --------------------------------------------------------------------------------------------

    # Random Target Parameter for Training & Test Sets Generation Functionality
    def label_gen(self):

        # Target & Test Label Index File Generation
        idxh_train = np.delete(np.arange(self.params.shape[0]), self.idxh_train)
        if self.settings.test_target_param != 0:
            idxh_test = idxh_train[np.sort(np.random.choice(len(idxh_train),
                                    self.settings.test_target_param, replace = False))]
            idxh_train = np.delete(idxh_train, np.where(np.in1d(idxh_train, idxh_test)))
            for i in range(self.settings.test_target_param): assert(idxh_test[i] not in idxh_train
                ), f"ERROR: Target Parameter #{i} for Training & Test Sets not mutually Exclusive"
            
            # Target & Test Label Index File Saving
            print(f">     Saving File Target Parameters for Training ({len(idxh_train)}) & Test {len(idxh_test)} Set's")
            np.savetxt(Path(f"{self.settings.datasave_folderpath}/Test Labels (V{self.settings.data_version}).txt"), idxh_test)
            np.savetxt(Path(f"{self.settings.datasave_folderpath}/Target Labels (V{self.settings.data_version}).txt"), idxh_train)
 
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
        mask = MUDI_cVNP.get_mask(settings, num_patient = num_patient)
        img = unmask(X.T, mask).get_fdata().T
        return torch.Tensor(img)

    # Patient Mask Retrieval Functionality
    def get_mask(settings, num_patient: int):
        mask_filepath = Path(f"{settings.mask_folderpath}/p{num_patient}.nii")
        assert(mask_filepath.exists()), f"ERROR: Patient {num_patient}'s Mask not Found!"
        #return torch.Tensor(np.array(load_img(mask_filepath).dataobj, dtype = np.float32))
        return load_img(mask_filepath)
