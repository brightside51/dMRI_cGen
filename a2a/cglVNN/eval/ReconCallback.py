# Library Imports
import sys
import io
import gc
import numpy as np
import argparse
import keras
import torch
import torch.nn as nn
import pytorch_lightning as pl
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt
import alive_progress
import time
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from pytorch_lightning.loggers import TensorBoardLogger
from nilearn.image import load_img
from nilearn.masking import unmask
from alive_progress import alive_bar
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_fcglVNN import MUDI_fcglVNN

# --------------------------------------------------------------------------------------------

# Fixed Mean 
class FixedMean(tf.keras.metrics.Mean):
    def update_state(self, y_true, y_pred, sample_weight = None):
        super().update_state(y_pred, sample_weight = sample_weight)

# Reconstruction Callback Class
class ReconCallback(keras.callbacks.Callback):
       
    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,              # Model Settings & Parametrizations
    ):

        # Class Variable Logging
        super().__init__()
        self.settings = settings; self.criterion = nn.MSELoss()

        # TensorBoard Logger Initialization
        self.train_logger = TensorBoardLogger(f'{self.settings.modelsave_folderpath}/V{self.settings.model_version}', 'train')
        self.val_logger = TensorBoardLogger(f'{self.settings.modelsave_folderpath}/V{self.settings.model_version}', 'validation')

    # --------------------------------------------------------------------------------------------

    # Patient Image Reconstruction Functionality
    def reconstruct(
        self,
        set: MUDI_fcglVNN,                  # Keras MUDI DataLoader
        X: np.ndarray,                      # Selected Patient Data
        mask: nib.nifti1.Nifti1Image,       # Selected Patient's Mask
        img: np.ndarray,                    # Selected Patient Image
        sel_slice: int = 25,                # Selected Reconstruction Slice
        epoch: int = 0                      # Epoch Number
    ):
        
        # Patient Image Reconstruction
        X_fake = self.model.predict_generator(set)
        best_train_loss = torch.ones(1, dtype = torch.float64) * 1000
        worst_train_loss = torch.zeros(1, dtype = torch.float64)
        best_val_loss = torch.ones(1, dtype = torch.float64) * 1000
        worst_val_loss = torch.zeros(1, dtype = torch.float64)
        
        # Voxel Reconstruction Loop for all Selected Target Parameters
        with alive_bar( len(set.idxh_recon),
                        title = f'Epoch {epoch} |' +
                        'Training Patient Reconstruction',
                        force_tty = True) as progress_bar:
            for i, param in enumerate(set.idxh_recon):

                # Selected Target Parameter Image Reconstruction
                X_param = X_fake[np.where(np.arange(len(X_fake)) % set.num_recon == i)[0]]
                X_gt = X[:, param].T.reshape((len(X_param), 1))
                loss = self.criterion(torch.Tensor(X_param), torch.Tensor(X_gt)); del X_gt
                
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
        return train_figure, best_train_loss, worst_train_loss, val_figure, best_val_loss, worst_val_loss

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the Start of Training
    def on_train_begin(self, logs = None):

        # Example Training Patient Download
        self.trainset = MUDI_fcglVNN(   self.settings, training = False,
                                        subject = self.settings.sel_train_patient)
        self.X_train, self.mask_train, self.img_train = MUDI_fcglVNN.get_patient(self.settings,
                                                            self.settings.sel_train_patient)

        # Example Validation Patient Download
        self.valset = MUDI_fcglVNN(     self.settings, training = False,
                                        subject = self.settings.sel_val_patient)
        self.X_val, self.mask_val, self.img_val = MUDI_fcglVNN.get_patient(  self.settings,
                                                            self.settings.sel_val_patient)

    # Functionality called upon the End of a Training Epoch
    def on_epoch_end(self, epoch, logs = None):

        # Epoch Update for Losses & Training Image Reconstruction
        tt_plot, tt_best_loss, tt_worst_loss, tv_plot, tv_best_loss, tv_worst_loss = self.reconstruct(  self.trainset, self.X_train,
                                                                                                        self.mask_train, self.img_train,
                                                                                                        sel_slice = 25, epoch = epoch)
        vt_plot, vt_best_loss, vt_worst_loss, vv_plot, vv_best_loss, vv_worst_loss = self.reconstruct(  self.valset, self.X_val,
                                                                                                        self.mask_val, self.img_val,
                                                                                                        sel_slice = 25, epoch = epoch)
        
        # TensorBoard Logger Model Visualizer, Update for Image Visualizer
        self.train_logger.experiment.add_scalar("Best Reconstruction Loss | Training", tt_best_loss, epoch)
        self.train_logger.experiment.add_scalar("Worst Reconstruction Loss | Training", tt_worst_loss, epoch)
        self.train_logger.experiment.add_scalar("Best Reconstruction Loss | Validation", tv_best_loss, epoch)
        self.train_logger.experiment.add_scalar("Worst Reconstruction Loss | Validation", tv_worst_loss, epoch)
        self.train_logger.experiment.add_figure("Training Image Reconstruction", tt_plot, epoch)
        self.train_logger.experiment.add_figure("Validation Image Reconstruction", tv_plot, epoch)

        self.val_logger.experiment.add_scalar("Best Reconstruction Loss | Training", vt_best_loss, epoch)
        self.val_logger.experiment.add_scalar("Worst Reconstruction Loss | Training", vt_worst_loss, epoch)
        self.val_logger.experiment.add_scalar("Best Reconstruction Loss | Validation", vv_best_loss, epoch)
        self.val_logger.experiment.add_scalar("Worst Reconstruction Loss | Validation", vv_worst_loss, epoch)
        self.val_logger.experiment.add_figure("Training Image Reconstruction", vt_plot, epoch)
        self.val_logger.experiment.add_figure("Validation Image Reconstruction", vv_plot, epoch)
