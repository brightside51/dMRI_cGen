# Library Imports
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##############################################################################################
# ----------------------------------------- Settings -----------------------------------------
##############################################################################################

# New Dataset Parametrizations Parser Initialization
if True:  
        data_parser = argparse.ArgumentParser(
                description = "1D MUDI Dataset Settings")
        data_parser.add_argument(                               # Dataset Version Variable
                '--version', type = int,                        # Default: 0
                default = 1,
                help = "Dataset Save Version")
        data_parser.add_argument(                               # Dataset Mode Selection Menu
                '--mode', type = str,                           # Default: Vertical
                default = 'h',
                choices = {'h', 'v', 'hv'},
                help = "Dataset Mode Selection Menu")
        data_parser.add_argument(                               # Dataset Batch Size Value
                '--batch_size', type = int,                     # Default: 500
                default = 1000,
                help = "Dataset Batch Size Value")

        # --------------------------------------------------------------------------------------------

        # Dataset Label Parametrization Arguments
        data_parser.add_argument(                               # Control Variable for the Inclusion of Patient ID in Labels
                '--patient_id', type = bool,            # Default: True
                default = False,
                help = "Control Variable for the Inclusion of Patient ID in Labels")
        data_parser.add_argument(                               # Control Variable for the Conversion of 3 Gradient Directions
                '--gradient_coord', type = bool,        # Coordinates into 2 Gradient Direction Angles (suggested by prof. Chantal)
                default = False,                        # Default: True (3 Coordinate Gradient Values)
                help = "Control Variable for the Conversion of Gradient Direction Mode")
        data_parser.add_argument(                               # Control Variable for the Rescaling & Normalization of Labels
                '--label_norm', type = bool,            # Default: True
                default = True,
                help = "Control Variable for the Rescaling & Normalization of Labels")
        data_settings = data_parser.parse_args("")
        num_labels = 7
        if not(data_settings.patient_id): num_labels -= 1       # Exclusion of Patiend ID
        if not(data_settings.gradient_coord): num_labels -= 1   # Conversion of Gradient Coordinates to Angles
        data_parser.add_argument(                               # Dataset Number of Labels
                '--num_labels', type = int,                         # Default: 7
                default = num_labels,
                help = "MUDI Dataset Number of Labels")

        # --------------------------------------------------------------------------------------------

        # Addition of File & Folderpath Arguments
        data_parser.add_argument(                               # Path for Main Dataset Folder
                '--main_folderpath', type = str,
                default = '../../../Datasets/MUDI Dataset',
                help = 'Main Folderpath for Root Dataset')
        data_settings = data_parser.parse_args("")
        data_parser.add_argument(                               # Path for Parameter Value File
                '--param_filepath', type = Path,
                default = Path(f'{data_settings.main_folderpath}/Raw Data/parameters_new.xlsx'),
                help = 'Input Filepath for Parameter Value Table')
        data_parser.add_argument(                               # Path for Parameter Value File
                '--data_filepath', type = Path,
                default = Path(f'{data_settings.main_folderpath}/Raw Data/data_.hdf5'),
                help = 'Input Filepath for Parameter Value Table')
        data_parser.add_argument(                               # Path for Patient Information File
                '--info_filepath', type = Path,
                default = Path(f'{data_settings.main_folderpath}/Raw Data/header_.csv'),
                help = 'Input Filepath for Patient Information Table')
        data_parser.add_argument(                               # Path for Folder Containing Mask Data Files
                '--mask_folderpath', type = Path,
                default = Path(f'{data_settings.main_folderpath}/Patient Mask'),
                help = 'Input Folderpath for Segregated Patient Mask Data')
        data_parser.add_argument(                               # Path for Dataset Saved Files
                '--save_folderpath', type = Path,
                default = Path(f'{data_settings.main_folderpath}/Saved Data/V{data_settings.version}'),
                help = 'Output Folderpath for MUDI Dataset Saved Versions')

        # --------------------------------------------------------------------------------------------

        # Dataset Splitting Arguments
        data_parser.add_argument(                               # List of Patients in MUDI Dataset
                '--patient_list', type = list,                  # Default: [11, 12, 13, 14, 15]
                default = [11, 12, 13, 14, 15],
                help = "List of Patients in MUDI Dataset")
        data_parser.add_argument(                               # Number of Patients to be used in the Test Set
                '--num_test_patients', type = int,              # Default: 1
                default = 2,
                help = "Number of Patients in Test Set")
        data_parser.add_argument(                               # Selected Test Patient
                '--sel_test_patients', type = int or list,      # Default: 14
                default = [14],
                help = "Selected Test Patient")
        data_parser.add_argument(                               # Number / Percentage of Parameters for Training Set's Training
                '--num_train_params', type = int,               # Default: 500
                default = 500,
                help = "Number / Percentage of Patients in the Training of the Training Set")

        # --------------------------------------------------------------------------------------------

        # Boolean Control Input & Shuffling Arguments
        data_parser.add_argument(                       # Ability to Shuffle the Patients that compose both Training and Test Sets
                '--patient_shuffle', type = bool,       # Default: False
                default = False,
                help = "Ability to Shuffle the Patients that compose both Training and Test Sets")
        data_parser.add_argument(                       # Ability to Shuffle the Samples inside both Training and Validation Sets
                '--sample_shuffle', type = bool,        # Default: False
                default = False,
                help = "Ability to Shuffle the Samples inside both Training and Validation Sets")
        data_parser.add_argument(                       # Number of Workers for DataLoader Usage
                '--num_workers', type = int,                # Default: 1
                default = 12,
                help = "Number of Workers for DataLoader Usage")
        data_settings = data_parser.parse_args("")

# Fully Connected Neural Network Model Parametrizations Parser Initialization
if True: 
        model_parser = argparse.ArgumentParser(
                description = "Voxel-Wise CVAE Settings")
        model_parser.add_argument(              # Model Version Variable
                '--model_version', type = int,  # Default: 0
                default = 0,
                help = "Experiment Version")
        model_parser.add_argument(              # Dataset Version Variable
                '--data_version', type = int,   # Default: 0
                default = 1,
                help = "MUDI Dataset Version")

        # --------------------------------------------------------------------------------------------

        # Addition of Filepath Arguments
        model_parser.add_argument(
                '--reader_folderpath', type = Path,
                default = Path(f'{data_settings.main_folderpath}/Dataset Reader'),
                help = 'Input Folderpath for MUDI Dataset Reader')
        model_parser.add_argument(
                '--data_folderpath', type = Path,
                default = Path(f'{data_settings.main_folderpath}/Saved Data/V{data_settings.version}'),
                help = 'Input Folderpath for MUDI Dataset Saved Versions')
        model_parser.add_argument(
                '--model_folderpath', type = str,
                default = 'Model Builds',
                help = 'Input Folderpath for Model Build & Architecture')
        model_parser.add_argument(
                '--script_folderpath', type = str,
                default = 'Training Scripts',
                help = 'Input Folderpath for Training & Testing Script Functions')
        model_parser.add_argument(
                '--save_folderpath', type = str,
                default = 'Saved Models',
                help = 'Output Folderpath for Saved & Saving Models')

        # --------------------------------------------------------------------------------------------

        # Addition of Training Requirement Arguments
        model_parser.add_argument(              # Number of Epochs
                '--num_epochs', type = int,     # Default: 1200
                default = 200,
                help = "Number of Epochs in Training Mode")
        model_settings = model_parser.parse_args("")
        model_parser.add_argument(              # Base Learning Rate
                '--base_lr', type = float,      # Default: 1e-4
                default = 1e-3,
                help = "Base Learning Rate Value in Training Mode")
        model_parser.add_argument(              # Weight Decay Value
                '--weight_decay', type = float, # Default: 1e-4
                default = 1e-4,
                help = "Weight Decay Value in Training Mode")
        model_parser.add_argument(              # Learning Rate Decay Ratio
                '--lr_decay', type = float,     # Default: 0.9
                default = 0.95,
                help = "Learning Rate Decay Value in Training Mode")
        model_parser.add_argument(              # Number of Equally Spaced Learning Rate Decay Epochs
                '--num_decay', type = int,      # Default: num_epochs / 10
                default = int(model_settings.num_epochs / 10),
                help = "Number of Equally Spaced Learning Rate Decay Epochs")
        model_parser.add_argument(              # Early Stopping Epoch Patience Value
                '--es_patience', type = int,    # Default: 1000 (no Early Stopping)
                default = 20,
                help = "Early Stopping Epoch Patience Value")
                
        # --------------------------------------------------------------------------------------------

        # Addition of Result Visualization Arguments
        model_parser.add_argument(                              # Selected Patient for Training Image Reconstruction
                '--sel_train_patient', type = int,              # Default: 11
                default = 11,
                help = "Selected Patient for Training Image Reconstruction")
        model_parser.add_argument(                              # Selected Patient for Validation Image Reconstruction
                '--sel_val_patient', type = int,                # Default: 15
                default = 14,
                help = "Selected Patient for Validation Image Reconstruction")
        model_parser.add_argument(                              # Selected Patient for Test Image Reconstruction
                '--sel_test_patient', type = int,               # Default: 14
                default = 14,
                help = "Selected Patient for Test Image Reconstruction")
        model_parser.add_argument(                              # Ability to Shuffle the Parameters chosen for Reconstruction
                '--recon_shuffle', type = bool,                 # Default: False
                default = False,
                help = "Ability to Shuffle the Parameters chosen for Reconstruction")
        model_parser.add_argument(                              # Selected Parameter for Training Image Reconstruction
                '--sel_train_param', type = int,                # Default: 14
                default = 300,
                help = "Selected Parameter for Training Image Reconstruction")
        model_parser.add_argument(                              # Selected Parameter for Validation Image Reconstruction
                '--sel_val_param', type = int,                  # Default: 14
                default = 500,
                help = "Selected Parameter for Validation Image Reconstruction")
        model_parser.add_argument('--data_settings', default = data_settings)

        # --------------------------------------------------------------------------------------------

        # Addition of Model Architecture Arguments
        model_parser.add_argument(              # Dataset Number of Labels
                '--num_labels', type = int,     # Default: 7
                default = data_settings.num_labels,
                help = "MUDI Dataset Number of Labels")
        model_parser.add_argument(              # Number of Parameter Settings for Training
                '--in_params', type = int,      # Default: 500
                default = data_settings.num_train_params,
                help = "Number of Parameter Settings for Training")
        model_parser.add_argument(              # Total Number of Parameter Settings
                '--out_params', type = int,     # Default: 1344
                default = 1344,
                help = "Total Number of Parameter Settings")
        model_parser.add_argument(              # Number of Hidden Layers in Neural Network
                '--num_hidden', type = int,     # Default: 2
                default = 2,
                help = "Number of Hidden Layers in Neural Network")

        model_settings = model_parser.parse_args("")
        model_settings.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################################
# ----------------------------------------- Dataset ------------------------------------------
##############################################################################################

# Dataset Access
sys.path.append(f"{data_settings.main_folderpath}/Dataset Reader")
from MUDI_1D import MUDI_1D
from SequenceMUDI import MRISelectorSubjDataset, MRIDecoderSubjDataset

# Dataset Version Creation
#data = MUDI_1D(data_settings)
#data.save()
"""
TrainLoader = MUDI_1D.loader(   Path(f"{model_settings.data_folderpath}"),
                                        version = model_settings.data_version,
                                        set_ = 'Train')
ValLoader = MUDI_1D.loader(   Path(f"{model_settings.data_folderpath}"),
                                        version = model_settings.data_version,
                                        set_ = 'Test')
print(len(TrainLoader)); print(len(ValLoader)); print(TrainLoader.batch_size)
"""

##############################################################################################
# --------------------------------------- fcNN Model -----------------------------------------
##############################################################################################

# Full fcNN All4One 2D VAE Model Class Importing
sys.path.append(model_settings.model_folderpath)
from fcNN import fcNN
sys.path.append(model_settings.script_folderpath)
from LitfcNN import LitfcNN

# --------------------------------------------------------------------------------------------

# Model Initialization & Training
fcNN = LitfcNN(model_settings)
fcNN_trainer = pl.Trainer(      max_epochs = model_settings.num_epochs,
                                devices = 1 if torch.cuda.is_available() else None,
                                accelerator = 'cuda', enable_progress_bar = True,
                                check_val_every_n_epoch = 1, num_sanity_val_steps = -1,
                                callbacks = [   TQDMProgressBar(refresh_rate = 1),
                                                EarlyStopping(  monitor = 'val_recon_loss', mode = 'min',
                                                                patience = model_settings.es_patience,
                                                                check_on_train_epoch_end = False),
                                                ModelCheckpoint(filename = f"{model_settings.save_folderpath}/V{model_settings.model_version}/fcNN (Best V{model_settings.model_version}).pth",
                                                                save_top_k = 1, save_on_train_epoch_end = True,
                                                                monitor = 'val_recon_loss', mode = 'min', verbose = True)])

fcNN_trainer.fit(fcNN)
#fcNN_trainer.test(fcNN)
#import tensorboard