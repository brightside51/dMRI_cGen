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
from nilearn.masking import unmask, apply_mask, compute_brain_mask, compute_background_mask
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_fcNN import MUDI_fcNN

# Full cglVNN Model Class Importing
sys.path.append("Model Builds")
from fcNN import fcNN
    
##############################################################################################
# ----------------------------------------- Settings -----------------------------------------
##############################################################################################

# [fcNN] Fully Connected Neural Network Model Parametrizations Parser Initialization
parser = argparse.ArgumentParser(
    description = "fcNN Model Settings")
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
    default = 1000,
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
parser.add_argument(                                    # Ability to Shuffle Target Voxels
    '--voxel_shuffle', type = bool,                     # Default: True
    default = True,
    help = "Ability to Shuffle Target Voxels")
parser.add_argument(                                    # Control Variable for the Sharing of Parameters inbetween Patient Sets
    '--interpatient_sharing', type = bool,              # Default: True (all Training Patient Sets will have, for the same epoch, ...
    default = False,                                    # ... the same exact parameters to reconstruct the data to, defined by the 1st Patient
    help = "Control Variable for the Sharing of Parameters inbetween Patient Sets")

# Dataset Settings | Subsectioning Arguments
parser.add_argument(                                    # Percentage of Used Target Voxels used in Training
    '--train_target_voxel', type = int or float,        # Default: 60%
    default = 100,
    help = "Percentage of Used Target Voxels used in Training")
parser.add_argument(                                    # Percentage of Used Target Voxels used in Validation
    '--val_target_voxel', type = int or float,          # Default: 100%
    default = 100,
    help = "Percentage of Used Target Voxels during Validation")
parser.add_argument(                                    # Selected Slice for Image Visualization
    '--sel_slice', type = int,                          # Default: 25
    default = 25,
    help = "Selected Slice for Image Visualization")

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
    default = 500,
    help = "MUDI Dataset No. of Training Parameters / Model's No. of Input Channels")
parser.add_argument(                    # Total Number of Parameter Settings
    '--out_channels', type = int,       # Default: 1344
    default = 1344,
    help = "Total Number of Parameter Settings")
parser.add_argument(                    # Number of Hidden Layers in Neural Network
    '--num_hidden', type = int,         # Default: 2
    default = 2,
    help = "Number of Hidden Layers in Neural Network")

settings = parser.parse_args(""); settings.device_ids = [3]
settings.device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")

##############################################################################################
# --------------------------------------- fcNN Model -----------------------------------------
##############################################################################################

# Seed Random State for Reproducibility
torch.manual_seed(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)

"""
# Model & Optimizer Setup
print(f"Evaluation\n     > Testing fcNN Model with {torch.cuda.device_count()} GPUs!")
model = fcNN(settings)#.to(settings.device)
model = nn.DataParallel(model, device_ids = settings.device_ids).to(settings.device)
optimizer = torch.optim.AdamW(  model.parameters(), lr = settings.base_lr,
                                weight_decay = settings.weight_decay)

# Model Checkpoint Loading
model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best fcNN.pt")
assert(model_filepath.exists()), f"ERROR: fcNN Model (V{settings.model_version}) not Found!"
checkpoint = torch.load(model_filepath, map_location = settings.device)
model.load_state_dict(checkpoint['Model'])
optimizer.load_state_dict(checkpoint['Optimizer'])
#save_epoch = checkpoint['Current Epoch']
torch.set_rng_state(checkpoint['RNG State']); del checkpoint
"""
save_epoch = 23#31#22
mode = 'Training'

# Experiment Logs Directories Initialization
mse_criterion = nn.MSELoss(reduction = 'none'); mse_criterion_mean = nn.MSELoss(reduction = 'mean')
mae_criterion = nn.L1Loss(reduction = 'none'); mae_criterion_mean = nn.L1Loss(reduction = 'mean')
checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs')
result_logger = TensorBoardLogger(checkpoint_folderpath, f'test/results/{mode}')
#mask_logger = TensorBoardLogger(checkpoint_folderpath, f'test/results/{mode}/mask')
print(f"     > Evaluating fcNN Model for Version #{settings.model_version}: {save_epoch} Past Epochs")

"""
# Image Sample Generation Test
test_set = MUDI_fcNN(       settings, subject = [settings.test_patient_list[0]],
                            target_voxel = 100)#settings.train_target_voxel)
test_loader = DataLoader(   dataset = test_set, pin_memory = False,
                            shuffle = False,#settings.train_sample_shuffle,
                            num_workers = 0,#settings.num_workers,
                            batch_size = len(test_set.idxv_target))
data_input = torch.Tensor(next(iter(test_loader))[0]); model.eval()
with torch.no_grad(): data_fake = torch.squeeze(model(data_input.to(settings.device)), dim = 1).T
print(np.all(data_gen[0, :] == data_fake.T[0, :]))
"""

# --------------------------------------------------------------------------------------------

# DataSet & DataLoader Initialization
test_set = MUDI_fcNN(       settings, subject = [settings.test_patient_list[0]],
                            target_voxel = 100)#settings.train_target_voxel)
idxh_target_filepath = Path(f"{settings.datasave_folderpath}/{mode} Labels (V{settings.data_version}).txt")
idxh_target = np.sort(np.loadtxt(idxh_target_filepath)).astype(int)
data_real = test_set.data[test_set.idxv_target][:, idxh_target].T

# Image Dataset Initialization
img_real = MUDI_fcNN.get_img(settings, num_patient = settings.test_patient_list[0])
img_gen = nib.load(os.path.join(    checkpoint_folderpath, f'test/epoch{save_epoch}/p{settings.test_patient_list[0]}/img_fake.nii.gz'))
mask = compute_background_mask(img_gen); data_gen = apply_mask(img_gen, mask).astype(np.float32)[idxh_target, :]
img_real = torch.Tensor(img_real)[idxh_target, :, :, :]; img_gen = torch.Tensor(np.array(img_gen.dataobj)).T[idxh_target, :, :, :]
#mse_img = np.empty((settings.out_channels, mask.shape[2], mask.shape[1], mask.shape[0]))
#ssim_img = np.empty((settings.out_channels, mask.shape[2], mask.shape[1], mask.shape[0]))
#mae_img = np.empty((settings.out_channels, mask.shape[2], mask.shape[1], mask.shape[0]))


# Scan Loop
data_range = (torch.max(img_real) - torch.min(img_real)).cpu().numpy()
scan_bar = tqdm(   enumerate(np.arange(len(idxh_target))), total = len(idxh_target),
    desc = f'Test Patient {settings.test_patient_list[0]}', unit = 'Scans')
for idx, scan_idx in scan_bar:

    # Loss Computation
    mse_img = mse_criterion(        img_gen[   scan_idx, :, :, :],
                                    img_real[  scan_idx, :, :, :]).detach().cpu().numpy()
    ssim_loss_mask, ssim_img = ssim(img_gen[   scan_idx, :, :, :].cpu().numpy().astype(np.float32),
                                    img_real[  scan_idx, :, :, :].cpu().numpy().astype(np.float32),
                                    full = True, win_size = 3, data_range = data_range)
    ssim_loss, _ = ssim(            img_gen[   scan_idx, :, :, :].cpu().numpy().astype(np.float32),
                                    img_real[  scan_idx, :, :, :].cpu().numpy().astype(np.float32),
        full = True, data_range = (torch.max(img_real[  scan_idx, :, :, :]) - torch.min(img_real[  scan_idx, :, :, :])).cpu().numpy())
    mae_img = mae_criterion(        img_gen[   scan_idx, :, :, :],
                                    img_real[  scan_idx, :, :, :]).detach().cpu().numpy()
    #mse_loss_mask = torch.mean(torch.Tensor(mse_img)); mae_loss_mask = torch.mean(torch.Tensor(mae_img))
    mse_loss = mse_criterion_mean(torch.Tensor(data_gen[scan_idx, :]), torch.Tensor(data_real[scan_idx, :]))
    mae_loss = mae_criterion_mean(torch.Tensor(data_gen[scan_idx, :]), torch.Tensor(data_real[scan_idx, :]))
    
    # --------------------------------------------------------------------------------------------
    
    # Target Parameter Plot Initialization
    scan_plot = plt.figure(figsize = (30, 25))
    plt.suptitle(f'Test Patient #{settings.test_patient_list[0]} | Parameter #{idxh_target[scan_idx]} | Slice #{settings.sel_slice}' +
                        f'\nMSE: {np.round(mse_loss.item(), 5)}| MAE: {np.round(mae_loss.item(), 5)} | SSIM: {np.round(ssim_loss, 5)}\n')
    plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
    gc.collect(); torch.cuda.empty_cache()
    
    # Original, Reconstruction & Loss Heatmap Plotting
    plt.subplot(2, 3, 1, title = 'Original Scan'); plt.imshow(img_real[scan_idx, settings.sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 2, title = 'Generated Scan'); plt.imshow(img_gen[scan_idx, settings.sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 4, title = 'MSE Loss Heatmap'); plt.imshow(mse_img[settings.sel_slice, :, :], cmap = 'hot')
    plt.subplot(2, 3, 5, title = 'MAE Loss Heatmap'); plt.imshow(mae_img[settings.sel_slice, :, :], cmap = 'hot')
    plt.subplot(2, 3, 6, title = 'SSIM Index Heatmap'); plt.imshow(ssim_img[settings.sel_slice, :, :], cmap = plt.cm.binary)

    # Loss Saving
    if len(idxh_target) > 1000 and scan_idx >= len(idxh_target) // 2:
        result_logger.experiment.add_scalar(f"MSE_Loss", mse_loss, scan_idx)
        #mask_logger.experiment.add_scalar(f"MSE_Loss", mse_loss_mask, scan_idx)
        result_logger.experiment.add_scalar(f"MAE_Loss", mae_loss, scan_idx)
        #mask_logger.experiment.add_scalar(f"MAE_Loss", mae_loss_mask, scan_idx)
        result_logger.experiment.add_scalar(f"SSIM_Index", ssim_loss, scan_idx)
        #mask_logger.experiment.add_scalar(f"SSIM_Index", ssim_loss_mask, scan_idx)
    else:
        result_logger.experiment.add_scalar(f"MSE Loss", mse_loss, scan_idx)
        #mask_logger.experiment.add_scalar(f"MSE Loss", mse_loss_mask, scan_idx)
        result_logger.experiment.add_scalar(f"MAE Loss", mae_loss, scan_idx)
        #mask_logger.experiment.add_scalar(f"MAE Loss", mae_loss_mask, scan_idx)
        result_logger.experiment.add_scalar(f"SSIM Index", ssim_loss, scan_idx)
        #mask_logger.experiment.add_scalar(f"SSIM Index", ssim_loss_mask, scan_idx)
    result_logger.experiment.add_figure(f"Target Results", scan_plot, scan_idx)
    # import tensorboard
