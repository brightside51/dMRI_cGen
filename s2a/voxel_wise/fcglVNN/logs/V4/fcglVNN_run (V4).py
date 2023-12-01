# Library Imports
import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import pytorch_lightning as pl
import h5py
import tqdm
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.preprocessing import StandardScaler
from nilearn.image import load_img
from nilearn.masking import unmask
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)
#print(os.environ["CUDA_VISIBLE_DEVICES"])
print(torch.cuda.device_count())
print(torch.cuda.is_available())
#os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'

##############################################################################################
# ----------------------------------------- Settings -----------------------------------------
##############################################################################################

# [fcglVNN] Fixed Conditional Generative Linear Voxel Neural Network Model Parametrizations Parser Initialization
parser = argparse.ArgumentParser(
    description = "fcglVNN Model Settings")
parser.add_argument(                            # Dataset Version Variable
    '--data_version', type = int,               # Default: 0
    default = 4,
    help = "Dataset Save Version")
parser.add_argument(                            # Model Version Variable
    '--model_version', type = int,              # Default: 0
    default = 4,
    help = "Experiment Version")
parser.add_argument(                            # Random Seed for Reproducibility
    '--seed', type = int,                       # Default: 42
    default = 42,
    help = "Random Seed for Reproducibility")

# --------------------------------------------------------------------------------------------

# Dataset Settings | Label Parametrization Arguments
parser.add_argument(                                    # Control Variable for the Inclusion of Patient ID in Labels
    '--patient_id', type = bool,                        # Default: True
    default = False,
    help = "Control Variable for the Inclusion of Patient ID in Labels")
parser.add_argument(                                    # Control Variable for the Conversion of 3 Gradient Directions
    '--gradient_coord', type = bool,                    # Coordinates into 2 Gradient Direction Angles (suggested by prof. Chantal)
    default = True,                                    # Default: True (3 Coordinate Gradient Values)
    help = "Control Variable for the Conversion of Gradient Direction Mode")
parser.add_argument(                                    # Control Variable for the Rescaling & Normalization of Labels
    '--label_norm', type = str,                         # Default: auto
    default = 'manual',
    choices = ['auto', 'manual', None],
    help = "Control Variable for the Rescaling & Normalization of Labels")
settings = parser.parse_args(""); num_labels = 7
if not(settings.patient_id): num_labels -= 1                                                # Exclusion of Patiend ID
if not(settings.gradient_coord) or settings.label_norm == 'manual': num_labels -= 1         # Conversion of Gradient Coordinates to Angles
parser.add_argument(                                                                        # Dataset Number of Labels
    '--num_labels', type = int,                                                             # Default: 7
    default = num_labels,
    help = "MUDI Dataset Number of Labels")

# --------------------------------------------------------------------------------------------

# Dataset Settings | Set Construction Arguments
parser.add_argument(                            # Complete List of Patients in Dataset
    '--patient_list', type = list,              # Default: [11, 12, 13, 14, 15]
    default = [11, 12, 13, 14, 15],
    help = "Complete List of Patients in Dataset")
parser.add_argument(                            # List of Patients used in Training
    '--train_patient_list', type = list,        # Default: [11, 12, 13]
    default = [11, 12, 13],
    help = "List of Patients used in Training")
parser.add_argument(                            # List of Patients used in Validation
    '--val_patient_list', type = list,          # Default: [15]
    default = [15],
    help = "List of Patients used in Validation")
parser.add_argument(                            # List of Patients used in Testing
    '--test_patient_list', type = list,         # Default: [14]
    default = [14],
    help = "List of Patients used in Testing")
parser.add_argument(                            # Number of Workers for DataLoader Usage
    '--num_workers', type = int,                # Default: 1
    default = 12,
    help = "Number of Workers for DataLoader Usage")
parser.add_argument(                            # DataLoader Batch Size
    '--batch_size', type = int,                 # Default: 250000
    default = 200000,
    help = "DataLoader Batch Size")

# --------------------------------------------------------------------------------------------

# Dataset Settings | Shuffling Arguments
parser.add_argument(                                    # Ability to Shuffle Training DataLoaders' Samples
    '--train_sample_shuffle', type = bool,              # Default: False
    default = True,
    help = "Ability to Shuffle Training DataLoaders' Samples")
parser.add_argument(                                    # Ability to Shuffle Validation DataLoaders' Samples
    '--val_sample_shuffle', type = bool,                # Default: False
    default = False,
    help = "Ability to Shuffle Validation DataLoaders' Samples")
parser.add_argument(                                    # Ability to Shuffle Target Parameters
    '--param_shuffle', type = bool,                     # Default: True
    default = True,
    help = "Ability to Shuffle Target Parameters")
parser.add_argument(                                    # Ability to Shuffle Target Voxels
    '--voxel_shuffle', type = bool,                     # Default: True
    default = True,
    help = "Ability to Shuffle Target Voxels")
parser.add_argument(                                    # Control Variable for the Inter-Patient Sharing of Target Voxels & Parameters
    '--interpatient_sharing', type = bool,              # Default: True (All Patients of the Same Set will have, for the Same Epoch, ...
    default = True,                                     # (... the Same Target Parameters & Voxels, defined by the 1st Patient)
    help = "Control Variable for the Inter-Patient Sharing of Target Voxels & Parameters")

# Dataset Settings | Subsectioning Arguments
parser.add_argument(                                    # Selected Slice for Image Visualization
    '--sel_slice', type = int,                          # Default: 25
    default = 25,
    help = "Selected Slice for Image Visualization")
parser.add_argument(                                    # Percentage of Used Target Parameters used in Training
    '--train_target_param', type = int or float,        # Default: 25% | 0.2% for 1 Target
    default = 100,
    help = "Percentage of Used Target Parameters used in Training")
parser.add_argument(                                    # Percentage of Used Target Parameters used in Validation
    '--val_target_param', type = int or float,          # Default: 5% | 0.9% for 2 Target
    default = 100,
    help = "Percentage of Used Target Parameters used in Validation")
parser.add_argument(                                    # Percentage of Used Target Voxels used in Training
    '--train_target_voxel', type = int or float,        # Default: 60%
    default = 60,
    help = "Percentage of Used Target Voxels used in Training")
parser.add_argument(                                    # Percentage of Used Target Voxels used in Validation
    '--val_target_voxel', type = int or float,          # Default: 100%
    default = 100,
    help = "Percentage of Used Target Voxels during Validation")
parser.add_argument(                                    # Number of Parameters used in the Test Set
    '--test_target_param', type = int,                  # Default: 100
    default = 0,
    help = "Number of Parameters used in the Test Set")

# --------------------------------------------------------------------------------------------

# Paths | Dataset-Related File & Folderpath Arguments
parser.add_argument(                                    # Path for Main Dataset Folder
    '--main_folderpath', type = str,
    default = '../../../Datasets/MUDI Dataset',
    help = 'Main Folderpath for Root Dataset')
settings = parser.parse_args("")
parser.add_argument(                                    # Path for Parameter Value File
    '--param_filepath', type = Path,
    default = Path(f'{settings.main_folderpath}/Raw Data/parameters_new.xlsx'),
    help = 'Input Filepath for Parameter Value Table')
parser.add_argument(                                    # Path for Parameter Value File
    '--data_filepath', type = Path,
    default = Path(f'{settings.main_folderpath}/Raw Data/data_.hdf5'),
    help = 'Input Filepath for Parameter Value Table')
parser.add_argument(                                    # Path for Patient Information File
    '--info_filepath', type = Path,
    default = Path(f'{settings.main_folderpath}/Raw Data/header_.csv'),
    help = 'Input Filepath for Patient Information Table')
parser.add_argument(                                    # Path for Folder Containing Mask Data Files
    '--mask_folderpath', type = Path,
    default = Path(f'{settings.main_folderpath}/Patient Mask'),
    help = 'Input Folderpath for Segregated Patient Mask Data')
parser.add_argument(                                    # Path for Dataset Saved Files
    '--datasave_folderpath', type = Path,
    default = Path(f'{settings.main_folderpath}/Saved Data/V{settings.data_version}'),
    help = 'Output Folderpath for MUDI Dataset Saved Versions')

# Paths | Model-Related File & Folderpath Arguments
parser.add_argument(                                    # Path for Dataset Reader Script
    '--reader_folderpath', type = Path,
    default = Path(f'{settings.main_folderpath}/Dataset Reader'),
    help = 'Input Folderpath for MUDI Dataset Reader')
parser.add_argument(                                    # Path for Model Build Script
    '--build_folderpath', type = str,
    default = 'Model Builds',
    help = 'Input Folderpath for Model Build & Architecture')
parser.add_argument(                                    # Path for Model Training Scripts
    '--script_folderpath', type = str,
    default = 'Training Scripts',
    help = 'Input Folderpath for Training & Testing Script Functions')
parser.add_argument(                                    # Path for Model Saved Files
    '--modelsave_folderpath', type = str,
    default = 'Saved Models',
    help = 'Output Folderpath for Saved & Saving Models')

# --------------------------------------------------------------------------------------------

# Model Settings | Optimization Arguments
parser.add_argument(                    # Number of Epochs
    '--num_epochs', type = int,         # Default: 1200
    default = 300,
    help = "Number of Epochs in Training Mode")
parser.add_argument(                    # Base Learning Rate
    '--base_lr', type = float,          # Default: 1e-4
    default = 1e-4,
    help = "Base Learning Rate Value in Training Mode")
parser.add_argument(                    # Weight Decay Value
    '--weight_decay', type = float,     # Default: 1e-4
    default = 1e-5,
    help = "Weight Decay Value in Training Mode")
parser.add_argument(                    # Learning Rate Decay Ratio
    '--lr_decay', type = float,         # Default: 0.9
    default = 0.9,
    help = "Learning Rate Decay Value in Training Mode")
parser.add_argument(                    # Early Stopping Epoch Patience Value
    '--es_patience', type = int,        # Default: 1000 (no Early Stopping)
    default = 5,
    help = "Early Stopping Epoch Patience Value")
parser.add_argument(                    # Early Stopping Delta Decay Value
    '--es_delta', type = float,         # Default: 1e-5
    default = 1e-5,
    help = "Early Stopping Delta Decay Value")

# --------------------------------------------------------------------------------------------

# Model Settings | Architecture Arguments
parser.add_argument(                    # Dataset Number of Training Parameters
    '--in_channels', type = int,        # Default: 500
    default = 100,
    help = "MUDI Dataset No. of Training Parameters / Model's No. of Input Channels")
parser.add_argument(                    # Number of Hidden Layers in Neural Network
    '--num_hidden', type = int,         # Default: 2
    default = 2,
    help = "Number of Hidden Layers in Neural Network")
parser.add_argument(                    # Hidden Layer's Top Number of Neurons
    '--top_hidden', type = int,         # Default: 64
    default = 1024,
    help = "Hidden Layer's Top Number of Neurons")
parser.add_argument(                    # Hidden Layer's Bottom Number of Neurons
    '--bottom_hidden', type = int,      # Default: 64
    default = 512,
    help = "Hidden Layer's Bottom Number of Neurons")

settings = parser.parse_args(""); settings.device_ids = [2]
settings.device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")

##############################################################################################
# ----------------------------------------- Dataset ------------------------------------------
##############################################################################################

# Dataset Access
#sys.path.append(settings.reader_folderpath)
#from MUDI_fcglVNN import MUDI_fcglVNN

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
            print(f">     Saving File Target Parameters for Training ({len(idxh_target)}) & Test {len(idxh_test)} Set's")
            np.savetxt(Path(f"{self.settings.datasave_folderpath}/Test Labels (V{self.settings.data_version}).txt"), idxh_test)
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

##############################################################################################
# -------------------------------------- fcglVNN Model ---------------------------------------
##############################################################################################

# Full cglVNN Model Training Class Importing
sys.path.append(settings.script_folderpath)
from fcglVNN_train import fcglVNN_train

# Model Training Method
gc.collect(); torch.cuda.empty_cache()
fcglVNN_train(settings)

#import tensorboard

##############################################################################################
# -------------------------------------- fcglVNN Test ----------------------------------------
##############################################################################################