# Library Imports
import os
import sys
import gc
import argparse
import torch
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
torch.autograd.set_detect_anomaly(True)
#os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

##############################################################################################
# ----------------------------------------- Settings -----------------------------------------
##############################################################################################

# [fcgCVAE] Fixed Conditional Generative Convolutional Variational AutoEncoder Model Parametrizations Parser Initialization
parser = argparse.ArgumentParser(
    description = "fcgCVAE Model Settings")
parser.add_argument(                            # Dataset Version Variable
    '--data_version', type = int,               # Default: 0
    default = 0,
    help = "Dataset Save Version")
parser.add_argument(                            # Model Version Variable
    '--model_version', type = int,              # Default: 0
    default = 0,
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
    default = 200,
    help = "DataLoader Batch Size")
parser.add_argument(                            # Final Input Image after Zero Padding Pre-Processing
    '--img_shape', type = int,                  # Default: 100
    default = 96,
    help = "Final Input Image after Zero Padding Pre-Processing")

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
parser.add_argument(                                    # Ability to Shuffle Target Slices
    '--slice_shuffle', type = bool,                     # Default: True
    default = True,
    help = "Ability to Shuffle Target Voxels")
parser.add_argument(                                    # Control Variable for the Sharing of Parameters inbetween Patient Sets
    '--interpatient_sharing', type = bool,              # Default: True (all Training Patient Sets will have, for the same epoch, ...
    default = False,                                    # ... the same exact parameters to reconstruct the data to, defined by the 1st Patient
    help = "Control Variable for the Sharing of Parameters inbetween Patient Sets")

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
    default = 60,
    help = "Percentage of Used Target Parameters used in Validation")
parser.add_argument(                                    # Number of Parameters used in the Test Set
    '--test_target_param', type = int,                  # Default: 100
    default = 0,
    help = "Number of Parameters used in the Test Set")

parser.add_argument(                                    # Percentage of Used Target Voxels used in Training
    '--train_target_slice', type = int or float,        # Default: 100%
    default = 60,
    help = "Percentage of Used Target Voxels used in Training")
parser.add_argument(                                    # Percentage of Used Target Voxels used in Validation
    '--val_target_slice', type = int or float,          # Default: 100%
    default = 100,
    help = "Percentage of Used Target Voxels during Validation")

parser.add_argument(                                    # Percentage of Used Source / Target Parameter Combos used in Training
    '--train_param_loop', type = int or float,          # Default: 60%
    default = 0,
    help = "Percentage of Used Target Voxels used in Training")
parser.add_argument(                                    # Percentage of Used Source / Target Parameter Combos used in Validation
    '--val_param_loop', type = int or float,            # Default: 60%
    default = 20,
    help = "Percentage of Used Target Voxels used in Training")

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
    default = 0.95,
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
    default = 500,
    help = "MUDI Dataset No. of Training Parameters / Model's No. of Input Channels")
parser.add_argument(                    # Number of Convolutional Blocks
    '--num_hidden', type = int,         # Default: 2
    default = 3,
    help = "Number of Convolutional Blocks")
parser.add_argument(                    # Latent Space Representation's Dimensionality
    '--dim_latent', type = int,         # Default: 128
    default = 128,
    help = "Latent Space Representation's Dimensionality")

parser.add_argument(                    # Hidden from Convolutional to Linear Sections Dimensionality
    '--dim_hidden', type = int,         # Default: 1024
    default = 1024,
    help = "Hidden from Convolutional to Linear Sections Dimensionality")
parser.add_argument(                    # Convolutional Blocks' Padding Value
    '--padding', type = int,            # Default: 1
    default = 2,
    help = "Convolutional Blocks' Padding Value")
parser.add_argument(                    # Convolutional Blocks' Kernel Size Value
    '--kernel_size', type = int,        # Default: 3
    default = 5,
    help = "Convolutional Blocks' Kernel Size Value")

settings = parser.parse_args(""); settings.device_ids = [1]
settings.device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

##############################################################################################
# ----------------------------------------- Dataset ------------------------------------------
##############################################################################################

# Dataset Access
#sys.path.append(settings.reader_folderpath)
#from MUDI_cVNP import MUDI_cVNP

##############################################################################################
# ------------------------------------- CVAE-GAN 2D Model ------------------------------------
##############################################################################################

# Full cglVNN Model Training Class Importing
sys.path.append(settings.script_folderpath)
from fcgCVAE_2D_train import fcgCVAE_2D_train

# Model Training Method
gc.collect(); torch.cuda.empty_cache()
fcgCVAE_2D_train(settings)
#tensorboard --logdir V0 --samples_per_plugin images=99999999 --bind_all

#import tensorboard

##############################################################################################
# ---------------------------------------- cVNP Test -----------------------------------------
##############################################################################################