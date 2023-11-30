# Library Imports
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import matplotlib.pyplot as plt
import time
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from nilearn.masking import unmask
from alive_progress import alive_bar
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_fcglVNN import MUDI_fcglVNN

# --------------------------------------------------------------------------------------------

# Reconstruction Callback Functionality
def ReconCallback(
    set: MUDI_fcglVNN,                  # Keras MUDI DataLoader
    X: torch.Tensor,                    # Selected Patient Data
    X_fake: torch.Tensor,               # Selected Patient Reconstruction
    mask: nib.nifti1.Nifti1Image,       # Selected Patient's Mask
    img: np.ndarray,                    # Selected Patient Image
    sel_slice: int = 25,                # Selected Reconstruction Slice
    epoch: int = 0                      # Epoch Number
):

    # Patient Image Reconstruction
    best_train_loss = torch.ones(1, dtype = torch.float64) * 1000
    worst_train_loss = torch.zeros(1, dtype = torch.float64)
    best_val_loss = torch.ones(1, dtype = torch.float64) * 1000
    worst_val_loss = torch.zeros(1, dtype = torch.float64)

    # Voxel Reconstruction Loop for all Selected Target Parameters
    with alive_bar( len(set.idxh_recon),
                    title = f'Epoch {epoch} | Reconstruction',
                    force_tty = True) as progress_bar:
        for i, param in enumerate(set.idxh_recon):

            # Selected Target Parameter Image Reconstruction
            X_param = X_fake[np.where(np.arange(len(X_fake)) % set.num_recon == i)[0]]
            #X_gt = X[np.where(np.arange(len(X)) % set.num_recon == i)[0]]
            X_gt = X[:, param].T.reshape((len(X_param), 1))
            loss = nn.MSELoss()(X_param, X_gt); del X_gt
            
            # Loss Computation for Training Parameter
            if param in set.idxh_train:
                if loss < best_train_loss:
                    best_train_loss = loss
                    best_train_idx = param
                    X_train_best = X_param
                if loss > worst_train_loss:
                    worst_train_loss = loss
                    worst_train_idx = param
                    X_train_worst = X_param

            # Loss Computation for Validation Parameter
            elif param in set.idxh_val:
                if loss < best_val_loss:
                    best_val_loss = loss
                    best_val_idx = param
                    X_val_best = X_param
                if loss > worst_val_loss:
                    worst_val_loss = loss
                    worst_val_idx = param
                    X_val_worst = X_param
            
            else: print("ERROR: Reconstruction Parameter not Found!")
            gc.collect(); torch.cuda.empty_cache()
            time.sleep(0.01); progress_bar()
    
    # --------------------------------------------------------------------------------------------

    # Training Image Unmasking of Original & Reconstructed Results
    X_train_best = unmask(X_train_best.T, mask).get_fdata().T
    X_train_worst = unmask(X_train_worst.T, mask).get_fdata().T

    # Training Example Original & Best Reconstructed Image Subplots
    train_figure = plt.figure(figsize = (20, 20))
    plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
    plt.subplot(2, 2, 1, title = f'Target Image (Parameter #{best_train_idx})')
    plt.imshow(img[best_train_idx, sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 2, title = f'Best Reconstruction (Parameter #{best_train_idx})')
    plt.imshow(X_train_best[0, sel_slice, :, :], cmap = plt.cm.binary)
    del best_train_idx, X_train_best
    
    # Training Example Original & Worst Reconstructed Image Subplots
    plt.subplot(2, 2, 3, title = f'Target Image (Parameter #{worst_train_idx})')
    plt.imshow(img[worst_train_idx, sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 4, title = f'Worst Reconstruction (Parameter #{worst_train_idx})')
    plt.imshow(X_train_worst[0, sel_slice, :, :], cmap = plt.cm.binary)
    del worst_train_idx, X_train_worst

    # --------------------------------------------------------------------------------------------

    # Training Image Unmasking of Original & Reconstructed Results
    X_val_best = unmask(X_val_best.T, mask).get_fdata().T
    X_val_worst = unmask(X_val_worst.T, mask).get_fdata().T

    # Training Example Original & Best Reconstructed Image Subplots
    val_figure = plt.figure(figsize = (20, 20))
    plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
    plt.subplot(2, 2, 1, title = f'Target Image (Parameter #{best_val_idx})')
    plt.imshow(img[best_val_idx, sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 2, title = f'Best Reconstruction (Parameter #{best_val_idx})')
    plt.imshow(X_val_best[0, sel_slice, :, :], cmap = plt.cm.binary)
    del best_val_idx, X_val_best
    
    # Training Example Original & Worst Reconstructed Image Subplots
    plt.subplot(2, 2, 3, title = f'Target Image (Parameter #{worst_val_idx})')
    plt.imshow(img[worst_val_idx, sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 4, title = f'Worst Reconstruction (Parameter #{worst_val_idx})')
    plt.imshow(X_val_worst[0, sel_slice, :, :], cmap = plt.cm.binary)
    del worst_val_idx, X_val_worst
    return {'Training Plot': train_figure,
            'Best Training Loss': best_train_loss,
            'Worst Training Loss': worst_train_loss,
            'Validation Plot': val_figure,
            'Best Validation Loss': best_val_loss,
            'Worst Validation Loss': worst_val_loss}
    