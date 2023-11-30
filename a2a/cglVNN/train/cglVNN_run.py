# Library Imports
import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import keras
import torch
import tensorflow as tf
import h5py
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
#from nilearn.masking import unmask
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from nilearn.image import load_img
from nilearn.masking import unmask
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

##############################################################################################
# ----------------------------------------- Settings -----------------------------------------
##############################################################################################

# [cglVNN] Conditional Generative Linear Voxel Neural Network Model Parametrizations Parser Initialization
if True:
        
        parser = argparse.ArgumentParser(
        description = "fcglVNN Model Settings")
        parser.add_argument(                            # Dataset Version Variable
        '--data_version', type = int,               # Default: 0
        default = 0,
        help = "Dataset Save Version")
        parser.add_argument(                            # Model Version Variable
        '--model_version', type = int,              # Default: 0
        default = 0,
        help = "Experiment Version")

        # --------------------------------------------------------------------------------------------

        # Dataset General Parametrization Arguments
        parser.add_argument(                            # Dataset Batch Size Value
        '--batch_size', type = int,                 # Default: 1000
        default = 50000,
        help = "Dataset Batch Size Value")
        parser.add_argument(                            # Number of Workers for DataLoader Usage
        '--num_workers', type = int,                # Default: 1
        default = 12,
        help = "Number of Workers for DataLoader Usage")
        parser.add_argument(                            # Ability to Shuffle the Samples inside both Training and Validation Sets
        '--sample_shuffle', type = bool,            # Default: False
        default = False,
        help = "Ability to Shuffle the Samples inside both Training and Validation Sets")
        parser.add_argument(                            # Number of Training Parameters
        '--num_train_params', type = int,           # Default: 500
        default = 500,
        help = "No. of Training Parameters")

        # --------------------------------------------------------------------------------------------

        # Dataset Label Parametrization Arguments
        parser.add_argument(                                    # Control Variable for the Inclusion of Patient ID in Labels
        '--patient_id', type = bool,                        # Default: True
        default = False,
        help = "Control Variable for the Inclusion of Patient ID in Labels")
        parser.add_argument(                                    # Control Variable for the Conversion of 3 Gradient Directions
        '--gradient_coord', type = bool,                    # Coordinates into 2 Gradient Direction Angles (suggested by prof. Chantal)
        default = False,                                    # Default: True (3 Coordinate Gradient Values)
        help = "Control Variable for the Conversion of Gradient Direction Mode")
        parser.add_argument(                                    # Control Variable for the Rescaling & Normalization of Labels
        '--label_norm', type = bool,                        # Default: True
        default = True,
        help = "Control Variable for the Rescaling & Normalization of Labels")
        settings = parser.parse_args(""); num_labels = 7
        if not(settings.patient_id): num_labels -= 1            # Exclusion of Patiend ID
        if not(settings.gradient_coord): num_labels -= 1        # Conversion of Gradient Coordinates to Angles
        parser.add_argument(                                    # Dataset Number of Labels
        '--num_labels', type = int,                         # Default: 7
        default = num_labels,
        help = "MUDI Dataset Number of Labels")

        # --------------------------------------------------------------------------------------------

        # Addition of Dataset-Related File & Folderpath Arguments
        parser.add_argument(                                    # Path for Main Dataset Folder
        '--main_folderpath', type = str,
        default = '../../Datasets/MUDI Dataset',
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

        # --------------------------------------------------------------------------------------------

        # Addition of Model-Related File & Folderpath Arguments
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

        # Addition of Training Requirement Arguments
        parser.add_argument(                    # Number of Epochs
        '--num_epochs', type = int,         # Default: 1200
        default = 300,
        help = "Number of Epochs in Training Mode")
        settings = parser.parse_args("")
        parser.add_argument(                    # Base Learning Rate
        '--base_lr', type = float,          # Default: 1e-4
        default = 1e-3,
        help = "Base Learning Rate Value in Training Mode")
        parser.add_argument(                    # Weight Decay Value
        '--weight_decay', type = float,     # Default: 1e-4
        default = 1e-4,
        help = "Weight Decay Value in Training Mode")
        parser.add_argument(                    # Learning Rate Decay Ratio
        '--lr_decay', type = float,         # Default: 0.9
        default = 0.9,
        help = "Learning Rate Decay Value in Training Mode")
        parser.add_argument(                    # Number of Epochs after which LR should Decay
        '--step_decay', type = int,         # Default: int(settings.num_epochs / 10)
        default = int(settings.num_epochs / 10),
        help = "Number of Epochs after which LR should Decay")
        parser.add_argument(                    # Early Stopping Epoch Patience Value
        '--es_patience', type = int,        # Default: 1000 (no Early Stopping)
        default = 50,
        help = "Early Stopping Epoch Patience Value")

        # --------------------------------------------------------------------------------------------

        # Addition of Model Architecture Arguments
        parser.add_argument(                    # Dataset Number of Training Parameters
        '--in_channels', type = int,        # Default: 500
        default = settings.num_train_params,
        help = "MUDI Dataset No. of Training Parameters / Model's No. of Input Channels")
        parser.add_argument(                    # Number of Hidden Layers in Neural Network
        '--num_hidden', type = int,         # Default: 2
        default = 2,
        help = "Number of Hidden Layers in Neural Network")
        parser.add_argument(                    # Hidden Layer's Top Number of Neurons
        '--top_hidden', type = int,         # Default: 64
        default = 64,
        help = "Hidden Layer's Top Number of Neurons")
        parser.add_argument(                    # Hidden Layer's Bottom Number of Neurons
        '--bottom_hidden', type = int,      # Default: 64
        default = 16,
        help = "Hidden Layer's Bottom Number of Neurons")

        # --------------------------------------------------------------------------------------------

        # Addition of Result Visualization Arguments
        parser.add_argument(                                    # Selected Patient for Training Image Reconstruction
        '--sel_train_patient', type = int,                  # Default: 11
        default = 11,
        help = "Selected Patient for Training Image Reconstruction")
        parser.add_argument(                                    # Selected Patient for Validation Image Reconstruction
        '--sel_val_patient', type = int,                    # Default: 15
        default = 15,
        help = "Selected Patient for Training Image Reconstruction")
        parser.add_argument(                                    # Selected Patient for Test Image Reconstruction
        '--sel_test_patient', type = int,                   # Default: 14
        default = 14,
        help = "Selected Patient for Test Image Reconstruction")
        parser.add_argument(                                    # Percentage of Reconstructed Parameters during Training
        '--param_recon_train', type = int or float,         # Default: 25% | 0.2% for 1 Reconstruction
        default = 25,
        help = "Percentage of Reconstructed Parameters during Training")
        parser.add_argument(                                    # Percentage of Reconstructed Parameters for Full Image Reconstruction
        '--param_recon_full', type = int or float,          # Default: 5% (20% of 25%) | 0.9% for 2 Reconstruction
        default = 5,
        help = "Percentage of Reconstructed Parameters for Full Image Reconstruction")
        parser.add_argument(                                    # Ability to Shuffle the Parameters for Reconstruction
        '--param_shuffle', type = bool,                     # Default: True
        default = True,
        help = "Ability to Shuffle the Parameters for Reconstruction")

        settings = parser.parse_args("")
        settings.device = torch.device('cuda:0') #)"cuda" if torch.cuda.is_available() else "cpu")

##############################################################################################
# ----------------------------------------- Dataset ------------------------------------------
##############################################################################################

# Dataset Access
#sys.path.append(settings.reader_folderpath)
#from MUDI_cglVNN import MUDI_cglVNN
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

# Dataset DataLoader Creation (Keras)
trainset = MUDI_cglVNN(settings, subject = [11, 12, 13])
valset = MUDI_cglVNN(settings, subject = [15])
#testset = MUDI_fcglVNN(settings, subject = [14])

##############################################################################################
# --------------------------------------- cglVNN Model ---------------------------------------
##############################################################################################

# Full cglVNN Model Class Importing
sys.path.append(settings.build_folderpath)
from cglVNN import fcglVNN
sys.path.append(settings.script_folderpath)
from ReconCallback import ReconCallback

# --------------------------------------------------------------------------------------------

# GPU Access Settings
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 2})
sess = tf.compat.v1.Session(config = config)
keras.backend.set_session(sess)
assert(len(tf.config.list_physical_devices('GPU')) >= 1), 'ERROR: No GPU Available!'
gc.collect(); torch.cuda.empty_cache()

# Callback Initialization
tensorboard_callback = keras.callbacks.TensorBoard(             log_dir = f"{settings.modelsave_folderpath}/V{settings.model_version}")
monitor_callback = keras.callbacks.ModelCheckpoint(             f"{settings.modelsave_folderpath}/V{settings.model_version}/cglVNN Best (V{settings.model_version})",
                                                                monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
earlystopping_callback = keras.callbacks.EarlyStopping(         monitor = 'val_loss', patience = settings.es_patience, mode = 'min')
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(   initial_learning_rate = settings.base_lr,
                                                                decay_steps = settings.step_decay * len(trainset),
                                                                decay_rate = settings.lr_decay)
recon_callback = ReconCallback(settings)

# Model Initialization
model = cglVNN(settings)
print(model.net.summary())
model.compile(  optimizer =     Adam(learning_rate = lr_schedule),
                loss =          'mean_squared_error')
"""model.compile(  optimizer =     Adam(learning_rate = lr_schedule),
                loss =          'mean_squared_error',
                metrics = [     FixedMean(name = 'Loss'),
                                FixedMean(name = 'Training Parameter Loss'),
                                FixedMean(name = 'Validation Parameter Loss')])"""

# Model Training
model.fit_generator(    trainset, validation_data = valset,
                        epochs = settings.num_epochs,
                        workers = settings.num_workers,
                        use_multiprocessing = False,
                        callbacks = [   tensorboard_callback, monitor_callback,
                                        recon_callback, earlystopping_callback])
model.save(f"{settings.modelsave_folderpath}/V{settings.model_version}/cglVNN (V{settings.model_version})")

#import tensorboard

##############################################################################################
# --------------------------------------- cglVNN Test ----------------------------------------
##############################################################################################