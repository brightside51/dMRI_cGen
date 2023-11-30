# Library Imports
import os
import sys
import gc
import sys
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import tqdm
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from nilearn.masking import unmask
from pytorch_lightning.loggers import TensorBoardLogger
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group
from sklearn.preprocessing import StandardScaler
from nilearn.image import load_img
from nilearn.masking import unmask
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_fcglVNN import MUDI_fcglVNN

# Full cglVNN Model Class Importing
sys.path.append("Model Builds")
from fcglVNN import fcglVNN
    
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

settings = parser.parse_args(""); settings.device_ids = [0]
settings.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

##############################################################################################
# -------------------------------------- fcglVNN Model ---------------------------------------
##############################################################################################

# Seed Random State for Reproducibility
torch.manual_seed(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)

# Model & Optimizer Setup
print(f"Evaluation\n     > Testing fcglVNN Model with {torch.cuda.device_count()} GPUs!")
mode = 'Train'; model = fcglVNN(settings)#.to(settings.device)
model = nn.DataParallel(model, device_ids = settings.device_ids).to(settings.device)
#optimizer = torch.optim.AdamW(  model.parameters(), lr = settings.base_lr,
#                                weight_decay = settings.weight_decay)

# Model Checkpoint Loading
model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best fcglVNN.pt")
assert(model_filepath.exists()), f"ERROR: fcglVNN Model (V{settings.model_version}) not Found!"
checkpoint = torch.load(model_filepath, map_location = settings.device)
model.load_state_dict(checkpoint['Model'])
#optimizer.load_state_dict(checkpoint['Optimizer'])
save_epoch = checkpoint['Current Epoch']
torch.set_rng_state(checkpoint['RNG State'])

# Experiment Logs Directories Initialization
mse_criterion = nn.MSELoss(reduction = 'mean'); mae_criterion = nn.L1Loss(reduction = 'mean'); del checkpoint
checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs')
quadrant_logger = TensorBoardLogger(checkpoint_folderpath, f'test/epoch{save_epoch}/p{settings.test_patient_list[0]}/{mode}/quadrant')
print(f"     > Evaluating fcglVNN Model for Version #{settings.model_version}: {save_epoch} Past Epochs")

# --------------------------------------------------------------------------------------------

# DataSet & DataLoader Initialization
idxh_target_filepath = Path(f"{settings.datasave_folderpath}/Target Labels (V{settings.data_version}).txt")
idxh_target = np.sort(np.loadtxt(idxh_target_filepath)).astype(int)
img_real = MUDI_fcglVNN.get_img(settings, num_patient = settings.test_patient_list[0])[idxh_target, :, :, :].T
img_gen = nib.load(os.path.join(checkpoint_folderpath, f'test/epoch{save_epoch}/p{settings.test_patient_list[0]}/{mode}/img_fake.nii.gz')).dataobj

# Quadrant Loop
hquadrant = 16; vquadrant = 23; num_quadrant = int((img_gen.shape[0] * img_gen.shape[1]) // (hquadrant * vquadrant))
quadrant_bar = tqdm(   enumerate(np.arange(num_quadrant)), total = num_quadrant,
    desc = f'Test Patient {settings.test_patient_list[0]}', unit = 'Quadrants')
for idx, quadrant_idx in quadrant_bar:

    # Quadrant Index Setting
    idxh = quadrant_idx % (img_gen.shape[0] // hquadrant)
    idxv = quadrant_idx // (img_gen.shape[0] // hquadrant)
    #rangeh = list([idxh * hquadrant : (idxh + 1) * hquadrant])
    #rangev = list([idxv * vquadrant : (idxv + 1) * vquadrant])
    #print(img_real[rangeh, rangev, :, :].shape)
    #print(torch.Tensor(img_gen[   idxh * hquadrant : (idxh + 1) * hquadrant, idxv * vquadrant : (idxv + 1) * vquadrant, :, :]).shape)

    # Loss Computation
    """
    mse_loss = mse_criterion(   img_real[   idxh * hquadrant : (idxh + 1) * hquadrant,
                                            idxv * vquadrant : (idxv + 1) * vquadrant, :, :],
                torch.Tensor(   img_gen[    idxh * hquadrant : (idxh + 1) * hquadrant,
                                            idxv * vquadrant : (idxv + 1) * vquadrant, :, :])).detach().cpu().numpy()
    ssim_loss, img_ssim = ssim( img_real[   idxh * hquadrant : (idxh + 1) * hquadrant,
                                            idxv * vquadrant : (idxv + 1) * vquadrant, :, :].T.cpu().numpy(), 
                torch.Tensor(   img_gen[    idxh * hquadrant : (idxh + 1) * hquadrant,
                                            idxv * vquadrant : (idxv + 1) * vquadrant, :, :]).T.cpu().numpy(), 
                                full = True, win_size = 3,
        data_range = (  torch.max(img_real[ idxh * hquadrant : (idxh + 1) * hquadrant,
                                            idxv * vquadrant : (idxv + 1) * vquadrant, :, :]) -\
                        torch.min(img_real[ idxh * hquadrant : (idxh + 1) * hquadrant,
                                            idxv * vquadrant : (idxv + 1) * vquadrant, :, :])).cpu().numpy())
    mae_loss = mae_criterion(   img_real[   idxh * hquadrant : (idxh + 1) * hquadrant,
                                            idxv * vquadrant : (idxv + 1) * vquadrant, :, :],
                torch.Tensor(   img_gen[    idxh * hquadrant : (idxh + 1) * hquadrant,
                                            idxv * vquadrant : (idxv + 1) * vquadrant, :, :])).detach().cpu().numpy()
    """

    # Loss Saving
    #quadrant_logger.experiment.add_scalar(f"MSE Loss", mse_loss, idx)
    #quadrant_logger.experiment.add_scalar(f"MAE Loss", mae_loss, idx)
    #quadrant_logger.experiment.add_scalar(f"SSIM Index", ssim_loss, idx)
    quadrant_logger.experiment.add_scalar(f"MSE Loss", np.mean(img_mse[    idxh * hquadrant : (idxh + 1) * hquadrant,
                                                                           idxv * vquadrant : (idxv + 1) * vquadrant, :, :]), idx)
    quadrant_logger.experiment.add_scalar(f"SSIM Index", np.mean(img_ssim[ idxh * hquadrant : (idxh + 1) * hquadrant, 
                                                                           idxv * vquadrant : (idxv + 1) * vquadrant, :, :]), idx)


