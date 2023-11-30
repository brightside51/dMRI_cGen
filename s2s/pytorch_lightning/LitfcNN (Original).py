# Library Imports
import sys
import io
import numpy as np
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import tensorflow as tf
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
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_1D import MUDI_1D
from SequenceMUDI import MRISelectorSubjDataset, MRIDecoderSubjDataset

# Full fcNN Model Class Importing
sys.path.append('../Model Builds')
from fcNN import fcNN

# --------------------------------------------------------------------------------------------

# fcNN Model Training, Validation & Testing Script Class
class LitfcNN(pl.LightningModule):

    ##############################################################################################
    # ---------------------------------------- Model Setup ---------------------------------------
    ##############################################################################################

    # Constructor / Initialization Function
    def __init__(
        self,
        settings: argparse.ArgumentParser,              # Model Settings & Parametrizations
    ):

        # Class Variable Logging
        super().__init__()
        self.settings = settings
        self.lr_decay_epochs = np.linspace( 0, self.settings.num_epochs,
                                            num = self.settings.num_decay).astype(int)[1::]

        # Model Initialization
        self.model = fcNN(                  in_params = self.settings.in_params,
                                            out_params = self.settings.out_params,
                                            num_hidden = self.settings.num_hidden)
        self.optimizer = torch.optim.Adam(  self.model.parameters(),
                                            lr = self.settings.base_lr,
                                            weight_decay = self.settings.weight_decay)
        self.criterion = nn.MSELoss(); self.past_epochs = 0

        # Existing Model Checkpoint Loading
        self.model_filepath = Path(f"{self.settings.save_folderpath}/V{self.settings.model_version}/fcNN (V{self.settings.model_version}).pth")
        #if self.settings.model_version != 0 and self.model_filepath.exists():
        if self.model_filepath.exists():

            # Checkpoint Fixing (due to the use of nn.DataParallel)
            print(f"DOWNLOADING Fully Connected Neural Network (Version {self.settings.model_version})")
            checkpoint = torch.load(self.model_filepath, map_location = self.settings.device); self.checkpoint_fix = dict()
            for sd, sd_value in checkpoint.items():
                if sd == 'ModelSD' or sd == 'OptimizerSD':
                    self.checkpoint_fix[sd] = OrderedDict()
                    for key, value in checkpoint[sd].items():
                        if key[0:7] == 'module.':
                            self.checkpoint_fix[sd][key[7:]] = value
                        else: self.checkpoint_fix[sd][key] = value
                else: self.checkpoint_fix[sd] = sd_value
            
            # Application of Checkpoint's State Dictionary
            self.model.load_state_dict(self.checkpoint_fix['ModelSD'])
            self.optimizer.load_state_dict(self.checkpoint_fix['OptimizerSD'])
            self.past_epochs = self.checkpoint_fix['Training Epochs']
            torch.set_rng_state(self.checkpoint_fix['RNG State'])
            del checkpoint
        self.lr_schedule = torch.optim.lr_scheduler.ExponentialLR(  self.optimizer,     # Learning Rate Decay
                                                    gamma = self.settings.lr_decay)     # in Chosen Epochs
        self.model = nn.DataParallel(self.model.to(self.settings.device))
        
    # Optimizer Initialization Function
    def configure_optimizers(self): return super().configure_optimizers()

    # Foward Functionality
    def forward(self, X): return self.model(X)

    ##############################################################################################
    # -------------------------------------- Dataset Setup ---------------------------------------
    ##############################################################################################
    
    # Train Set DataLoader Download
    def train_dataloader(self):
        TrainLoader = MUDI_1D.loader(   Path(f"{self.settings.data_folderpath}"),
                                        version = self.settings.data_version,
                                        set_ = 'Train')
        self.train_batches = len(TrainLoader)
        return TrainLoader

    # Test Set DataLoader Download
    def test_dataloader(self):
        TestLoader = MUDI_1D.loader(    Path(f"{self.settings.data_folderpath}"),
                                        version = self.settings.data_version,
                                        set_ = 'Test')
        self.test_batches = len(TestLoader)
        return TestLoader

    # --------------------------------------------------------------------------------------------

    # Patient Image Reconstruction
    def reconstruct(
        self,
        pX: torch.Tensor,               # Selected Patient's Data
        pMask: torch.Tensor,            # Selected Patient's Mask
        pX_img: torch.Tensor,           # Selected Patient Image
        sel_slice: int = 25             # Selected Reconstruction Slice
    ):
        
        # Fake 3D Image Generation
        with torch.no_grad(): pX_fake = self.model(pX[self.data.idxh_train, :].T).detach().cpu()
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

        # Validation Parameter Example Original & Reconstructed Image Subplot
        val_figure = plt.figure(figsize = (20, 10))
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
        plt.subplot(1, 2, 1, title = f'Original Image (Parameter #{self.sel_val_param})')
        plt.imshow(pX_img[self.sel_val_param, sel_slice, :, :], cmap = plt.cm.binary)
        plt.subplot(1, 2, 2, title = f'Reconstructed Image (Parameter #{self.sel_val_param})')
        plt.imshow(pX_fake[self.sel_val_param, sel_slice, :, :], cmap = plt.cm.binary)
        return recon_loss, train_figure, val_figure
    
    ##############################################################################################
    # ------------------------------------- Training Script --------------------------------------
    ##############################################################################################

    # Functionality called upon the Start of Training
    def on_train_start(self):
        
        # Model Training Mode Setup
        self.model.train()
        self.automatic_optimization = False

        # Example Training & Validation Patients Download
        self.data = MUDI_1D(self.settings.data_settings); self.best_val_loss = 1
        self.pX_train, self.pMask_train = self.data.get_patient(self.settings.sel_train_patient)
        self.pX_train_img = unmask(self.pX_train, self.pMask_train).get_fdata().T
        self.pX_val, self.pMask_val = self.data.get_patient(self.settings.sel_val_patient)
        self.pX_val_img = unmask(self.pX_val, self.pMask_val).get_fdata().T
        
        # Reconstruction Training & Validation Parameter Definition
        if not self.settings.recon_shuffle:
            self.sel_train_param = self.data.idxh_train[self.settings.sel_train_param]
            self.sel_val_param = self.data.idxh_val[self.settings.sel_val_param]

        # TensorBoard Logger Initialization
        self.train_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Training Performance')
        self.val_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Validation Performance')
        #self.train_logger.experiment.add_graph(self.model, torch.rand(1, self.settings.in_params).to(self.settings.device))

    # Functionality called upon the Start of Training Epoch
    def on_train_epoch_start(self): self.train_loss = 0; self.model.train()

    # --------------------------------------------------------------------------------------------

    # Training Step / Batch Loop 
    def training_step(self, batch, batch_idx):

        # Dataset Comparison Mode
        #X_train_batch, X_val_batch = batch
        #X_batch = torch.cat((X_train_batch, X_val_batch), 1)
        #print(np.where(torch.Tensor(self.trainset[batch_idx][0]) == X_train_batch.detach().cpu()) == False)

        # Data Handling
        X_train_batch, X_val_batch = batch
        X_train_batch = X_train_batch.type(torch.float).to(self.settings.device)
        X_batch = torch.cat((X_train_batch, X_val_batch), 1).type(torch.float).to(self.settings.device)
        X_batch = X_batch.type(torch.float).to(self.settings.device)

        # Forward Propagation & Loss Computation
        X_fake_batch = self.model(X_train_batch)
        loss = self.criterion(X_fake_batch, X_batch)
        #train_loss = Variable(loss, requires_grad = True)

        # Backwards Propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del X_batch, X_train_batch, X_val_batch, X_fake_batch
        return loss

    # Functionality called upon the End of a Batch Training Step
    def on_train_batch_end(self, loss, batch, batch_idx): self.train_loss = self.train_loss + loss['loss'].item()

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Training Epoch
    def on_train_epoch_end(self):

        # Learning Rate Decay
        self.model.eval()
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # Epoch Update for Losses & Training Image Reconstruction
        self.num_epochs = self.past_epochs + self.current_epoch
        self.train_loss = self.train_loss / self.train_batches
        train_recon_loss, tt_plot, tv_plot = self.reconstruct(  self.pX_train, self.pMask_train,
                                                                self.pX_train_img, sel_slice = 25)
        val_recon_loss, vt_plot, vv_plot = self.reconstruct(    self.pX_val, self.pMask_val,
                                                                self.pX_val_img, sel_slice = 25)
        self.log('val_recon_loss', val_recon_loss)
        train_recon_loss = Variable(train_recon_loss, requires_grad = True)
        self.optimizer.zero_grad(); train_recon_loss.backward(); self.optimizer.step()
        
        # TensorBoard Logger Model Visualizer, Update for Scalar Values & Image Visualizer
        self.train_logger.experiment.add_scalar("Loss", self.train_loss, self.num_epochs)
        self.train_logger.experiment.add_scalar("Reconstruction Loss", train_recon_loss, self.num_epochs)
        self.train_logger.experiment.add_figure("Training Image Reconstruction", tt_plot, self.num_epochs)
        self.train_logger.experiment.add_figure("Validation Image Reconstruction", tv_plot, self.num_epochs)
        self.val_logger.experiment.add_scalar("Reconstruction Loss", val_recon_loss, self.num_epochs)
        self.val_logger.experiment.add_figure("Training Image Reconstruction", vt_plot, self.num_epochs)
        self.val_logger.experiment.add_figure("Validation Image Reconstruction", vv_plot, self.num_epochs)

        # Model Best Checkpoint Saving
        if val_recon_loss <= self.best_val_loss:
            self.best_val_loss = val_recon_loss
            torch.save({'ModelSD': self.model.state_dict(),
                        'OptimizerSD': self.optimizer.state_dict(),
                        'Training Epochs': self.num_epochs,
                        'RNG State': torch.get_rng_state()},
                        self.model_filepath)
        del train_recon_loss, tt_plot, tv_plot, val_recon_loss, vt_plot, vv_plot
            
    ##############################################################################################
    # -------------------------------------- Testing Script --------------------------------------
    ##############################################################################################

    

