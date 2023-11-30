# Library Imports
import sys
import io
import numpy as np
import argparse
import keras
import torch
import torch.nn as nn
import pytorch_lightning as pl
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from pytorch_lightning.loggers import TensorBoardLogger
from nilearn.image import load_img
from nilearn.masking import unmask
from PIL import Image
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_1D import MUDI_1D
from SequenceMUDI import MRISelectorSubjDataset, MRIDecoderSubjDataset

# --------------------------------------------------------------------------------------------

# Reconstruction Callback Class
class ReconCallback(keras.callbacks.Callback):
       
    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,              # Model Settings & Parametrizations
    ):

        # Class Variable Logging
        super().__init__()
        self.settings = settings
        self.data = MUDI_1D(self.settings.data_settings); self.criterion = nn.MSELoss()

        # TensorBoard Logger Initialization
        self.train_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'train')
        self.val_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'validation')
        #self.train_log = f"{self.settings.save_folderpath}/V{self.settings.model_version}/Training Performance/train"
        #self.val_log = f"{self.settings.save_folderpath}/V{self.settings.model_version}/Training Performance/val"
        #self.train_writer = tf.summary.create_file_writer(self.train_log)
        #self.val_writer = tf.summary.create_file_writer(self.val_log)

    # --------------------------------------------------------------------------------------------

    # Patient Image Reconstruction
    def reconstruct(
        self,
        set: MRIDecoderSubjDataset,     # Keras DataLoader
        pMask: torch.Tensor,            # Selected Patient's Mask
        pX_img: torch.Tensor,           # Selected Patient Image
        sel_slice: int = 25             # Selected Reconstruction Slice
    ):
        
        # Fake 3D Image Generation
        pX_fake = self.model.predict_generator(set)
        pX_fake = unmask(pX_fake.T, pMask).get_fdata().T
        assert(np.all(pX_img.shape == pX_fake.shape)), "ERROR: Unmasking went Wrong!"
        recon_loss = self.criterion(torch.Tensor(pX_fake), torch.Tensor(pX_img))

        # Randomly Selected Training & Validation Parameters for Visualization
        if self.settings.recon_shuffle:
            self.sel_train_param = np.random.choice(self.data.idxh_train, size = 1, replace = False)[0]
            self.sel_val_param = np.random.choice(self.data.idxh_val, size = 1, replace = False)[0]

        # --------------------------------------------------------------------------------------------
        
        # Training Parameter Example Original & Reconstructed Image Subplot
        train_figure = plt.figure(figsize = (20, 10))
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
        plt.subplot(1, 2, 1, title = f'Original Image (Parameter #{self.sel_train_param})')
        plt.imshow(pX_img[self.sel_train_param, sel_slice, :, :], cmap = plt.cm.binary)
        plt.subplot(1, 2, 2, title = f'Reconstructed Image (Parameter #{self.sel_train_param})')
        plt.imshow(pX_fake[self.sel_train_param, sel_slice, :, :], cmap = plt.cm.binary)
        #img_buffer = io.BytesIO(); plt.savefig(img_buffer, format = 'png')
        #train_figure = Image.open(img_buffer); img_buffer.close()

        # Validation Parameter Example Original & Reconstructed Image Subplot
        val_figure = plt.figure(figsize = (20, 10))
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
        plt.subplot(1, 2, 1, title = f'Original Image (Parameter #{self.sel_val_param})')
        plt.imshow(pX_img[self.sel_val_param, sel_slice, :, :], cmap = plt.cm.binary)
        plt.subplot(1, 2, 2, title = f'Reconstructed Image (Parameter #{self.sel_val_param})')
        plt.imshow(pX_fake[self.sel_val_param, sel_slice, :, :], cmap = plt.cm.binary)
        #img_buffer = io.BytesIO(); plt.savefig(img_buffer, format = 'png')
        #val_figure = Image.open(img_buffer); img_buffer.close()
        return recon_loss, train_figure, val_figure

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the Start of Training
    def on_train_begin(self, logs = None):

        # Example Training Patient Download 
        self.pX_train, self.pMask_train = self.data.get_patient(self.settings.sel_train_patient)
        self.pX_train_img = unmask(self.pX_train, self.pMask_train).get_fdata().T
        self.train_set = MRIDecoderSubjDataset(  root_dir = f'{self.settings.data_settings.main_folderpath}/Raw Data/',
                                                selecf = f'{self.settings.data_folderpath}/1D Training Labels (V{self.settings.data_version}).txt',
                                                dataf = 'data_.hdf5', headerf = 'header_.csv',
                                                subj_list = np.array([self.settings.sel_train_patient]),
                                                batch_size = self.settings.data_settings.batch_size)
        
        # Example Validation Patient Download
        self.pX_val, self.pMask_val = self.data.get_patient(self.settings.sel_val_patient)
        self.pX_val_img = unmask(self.pX_val, self.pMask_val).get_fdata().T
        self.val_set = MRIDecoderSubjDataset(  root_dir = f'{self.settings.data_settings.main_folderpath}/Raw Data/',
                                                selecf = f'{self.settings.data_folderpath}/1D Training Labels (V{self.settings.data_version}).txt',
                                                dataf = 'data_.hdf5', headerf = 'header_.csv',
                                                subj_list = np.array([self.settings.sel_val_patient]),
                                                batch_size = self.settings.data_settings.batch_size)

        # Reconstruction Training & Validation Parameter Definition
        if not self.settings.recon_shuffle:
            self.sel_train_param = self.data.idxh_train[self.settings.sel_train_param]
            self.sel_val_param = self.data.idxh_val[self.settings.sel_val_param]
    
    # Functionality called upon the End of a Training Epoch
    def on_epoch_end(self, epoch, logs = None):

        # Randomly Selected Training & Validation Parameters for Visualization
        if self.settings.recon_shuffle:
            self.sel_train_param = np.random.choice(self.data.idxh_train, size = 1, replace = False)[0]
            self.sel_val_param = np.random.choice(self.data.idxh_val, size = 1, replace = False)[0]
        
        # Epoch Update for Losses & Training Image Reconstruction
        train_recon_loss, tt_plot, tv_plot = self.reconstruct(  self.train_set, self.pMask_train,
                                                                self.pX_train_img, sel_slice = 25)
        val_recon_loss, vt_plot, vv_plot = self.reconstruct(    self.val_set, self.pMask_val,
                                                                self.pX_val_img, sel_slice = 25)
        
        # TensorBoard Logger Model Visualizer, Update for Image Visualizer
        self.train_logger.experiment.add_scalar("Reconstruction Loss", train_recon_loss, epoch)
        self.train_logger.experiment.add_figure("Training Image Reconstruction", tt_plot, epoch)
        self.train_logger.experiment.add_figure("Validation Image Reconstruction", tv_plot, epoch)
        self.val_logger.experiment.add_scalar("Reconstruction Loss", val_recon_loss, epoch)
        self.val_logger.experiment.add_figure("Training Image Reconstruction", vt_plot, epoch)
        self.val_logger.experiment.add_figure("Validation Image Reconstruction", vv_plot, epoch)

        """
        # TensorBoard Logger Model Visualizer, Update for Image Visualizer
        with self.train_writer.as_default():
            tf.summary.scalar('Reconstruction Loss', train_recon_loss, step = epoch)
            tf.summary.image('Training Image Reconstruction', tt_plot, step = epoch)
            tf.summary.image('Training Image Reconstruction', tv_plot, step = epoch)
        self.train_writer.flush()
        with self.val_writer.as_default():
            tf.summary.scalar('Reconstruction Loss', val_recon_loss, step = epoch)
            tf.summary.image('Training Image Reconstruction', vt_plot, step = epoch)
            tf.summary.image('Training Image Reconstruction', vv_plot, step = epoch)
        self.train_writer.flush()
        """

