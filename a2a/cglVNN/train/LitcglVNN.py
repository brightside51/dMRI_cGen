# Library Imports
import sys
import io
import numpy as np
import pandas as pd
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
from pytorch_lightning.loggers import TensorBoardLogger
from nilearn.image import load_img
from nilearn.masking import unmask
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_1D import MUDI_1D

# Full fcNN Model Class Importing
sys.path.append('../Model Builds')
from cglVNN import cglVNN

# --------------------------------------------------------------------------------------------

# cglVNN Model Training, Validation & Testing Script Class
class LitcglVNN(pl.LightningModule):

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
        self.lr_decay_epochs = [80, 140]                # Epochs for Learning Rate Decay

        # Model Initialization
        self.model = cglVNN(                num_labels = self.settings.num_labels,
                                            num_hidden = self.settings.num_hidden,
                                            var_hidden = self.settings.var_hidden)
        self.optimizer = torch.optim.Adam(  self.model.parameters(),
                                            lr = self.settings.base_lr,
                                            weight_decay = self.settings.weight_decay)
        self.criterion = nn.MSELoss(); self.past_epochs = 0

        # Existing Model Checkpoint Loading
        self.model_filepath = Path(f"{self.settings.save_folderpath}/V{self.settings.model_version}/cglVNN (V{self.settings.model_version}).pth")
        if self.settings.model_version != 0 and self.model_filepath.exists():

            # Checkpoint Fixing (due to the use of nn.DataParallel)
            print(f"DOWNLOADING Conditional Generative Linear Voxel Neural Network (Version {self.settings.model_version})")
            checkpoint = torch.load(self.model_filepath); self.checkpoint_fix = dict()
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
        self.model = self.model.to(self.settings.device)    # nn.DataParallel(self.model).to(self.settings.device))
        
    # Optimizer Initialization Function
    def configure_optimizers(self): return super().configure_optimizers()

    # Foward Functionality
    def forward(self, X_train, y_train, y_val): return self.model(X_train, y_train, y_val)

    ##############################################################################################
    # -------------------------------------- Dataset Setup ---------------------------------------
    ##############################################################################################
    
    # Train Set DataLoader Download
    def train_dataloader(self):
        TrainTrainLoader = MUDI_1D.loader(  Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            set_ = 'Test', mode_ = 'Train')
        TrainValLoader = MUDI_1D.loader(    Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            set_ = 'Test', mode_ = 'Val')
        assert(len(TrainTrainLoader) == len(TrainValLoader)
               ), f"ERROR: DataLoaders wrongly built!"
        self.train_batches = len(TrainTrainLoader)
        self.val_batches = len(TrainValLoader)
        return {'train': TrainTrainLoader, 'val': TrainValLoader}

    # Test Set DataLoader Download
    def test_dataloader(self):
        TestTrainLoader = MUDI_1D.loader(   Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            set_ = 'Test', mode_ = 'Train')
        TestValLoader = MUDI_1D.loader(     Path(f"{self.settings.data_folderpath}"),
                                            version = self.settings.data_version,
                                            set_ = 'Test', mode_ = 'Val')
        assert(len(TestTrainLoader) == len(TestValLoader)
               ), f"ERROR: DataLoaders wrongly built!"
        self.test_batches = len(TestTrainLoader)
        return {'train': TestTrainLoader, 'val': TestValLoader}

    # --------------------------------------------------------------------------------------------

    # Patient Image Reconstruction
    def reconstruct(
        self,
        #sel_train_param: int = 300,     # Selected Parameter Setting Index for Training Image Reconstruction
        #sel_val_param: int = 1000,      # Selected Parameter Setting Index for Validation Image Reconstruction
        sel_slice: int = 25,            # Selected Slice Index
    ):
        
        # Voxel Reconstruction Loop (all Parameters)
        best_train_loss = torch.ones(1); worst_train_loss = torch.zeros(1)
        best_val_loss = torch.ones(1); worst_val_loss = torch.zeros(1)
        for p in self.idx_recon_batch:
            
            # Patient Image's Current Parameter Combination
            pX_real = self.pX[:, p].T.view(self.pX.shape[0], 1)
            py_real = torch.Tensor(self.data.params.iloc[p, :].values)
            py_real = py_real.repeat(pX_real.shape[0], 1)

            # Selected Training Parameter Fake Image Generation
            pX_train = self.model(pX_real, py_real, self.py_train)
            p_train_loss = self.criterion(pX_train, self.pX[:, self.settings.sel_train_param].T)
            if p_train_loss < best_train_loss: best_train_idx = p; best_train_loss = p_train_loss; pX_train_best = pX_train
            if p_train_loss > worst_train_loss: worst_train_idx = p; worst_train_loss = p_train_loss; pX_train_worst = pX_train

            # Selected Validation Parameter Fake Image Generation
            pX_val = self.model(pX_real, py_real, self.py_val)
            p_val_loss = self.criterion(pX_val, self.pX[:, self.settings.sel_val_param].T)
            if p_val_loss < best_val_loss: best_val_idx = p; best_val_loss = p_val_loss; pX_val_best = pX_val
            if p_val_loss > worst_val_loss: worst_val_idx = p; worst_val_loss = p_val_loss; pX_val_worst = pX_val

        # Original & Fake Image Unmasking
        pX_train_gt = unmask(self.pX[:, self.settings.sel_train_param], self.pMask).get_fdata().T
        pX_train_best = unmask(pX_train_best.detach().numpy().T, self.pMask).get_fdata().T
        pX_train_worst = unmask(pX_train_worst.detach().numpy().T, self.pMask).get_fdata().T
        pX_val_gt = unmask(self.pX[:, self.settings.sel_val_param], self.pMask).get_fdata().T
        pX_val_best = unmask(pX_val_best.detach().numpy().T, self.pMask).get_fdata().T
        pX_val_worst = unmask(pX_val_worst.detach().numpy().T, self.pMask).get_fdata().T
        del pX_train, pX_val, p_train_loss, pX_real, py_real

        # --------------------------------------------------------------------------------------------

        # Training Example Original & Reconstructed Image Subplots
        train_figure = plt.figure(self.num_epochs, figsize = (30, 20))
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
        plt.title(f'Epoch #{self.num_epochs} | Patient #{self.settings.sel_train_patient}'
        + f' | Training Parameter Combo #{self.settings.sel_train_param} | Slice #{sel_slice}')
        
        plt.subplot(3, 1, 1, title = f'Target Image (Parameter #{self.settings.sel_train_param})')
        plt.imshow(pX_train_gt[sel_slice, :, :], cmap = plt.cm.binary)
        plt.subplot(3, 1, 2, title = f'Best Reconstruction (Parameter #{best_train_idx})')
        plt.imshow(pX_train_best[0, sel_slice, :, :], cmap = plt.cm.binary)
        plt.subplot(3, 1, 3, title = f'Worst Reconstruction (Parameter #{worst_train_idx})')
        plt.imshow(pX_train_worst[0, sel_slice, :, :], cmap = plt.cm.binary)

        # --------------------------------------------------------------------------------------------

        # Training Example Original & Reconstructed Image Subplots
        val_figure = plt.figure(self.num_epochs, figsize = (30, 30))
        plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
        plt.title(f'Epoch #{self.num_epochs} | Patient #{self.settings.sel_train_patient}'
        + f' | Validation Parameter Combo #{self.settings.sel_val_param} | Slice #{sel_slice}')
        
        plt.subplot(3, 1, 1, title = f'Target Image (Parameter #{self.settings.sel_val_param})')
        plt.imshow(pX_val_gt[sel_slice, :, :], cmap = plt.cm.binary)
        plt.subplot(3, 1, 2, title = f'Best Reconstruction (Parameter #{best_val_idx})')
        plt.imshow(pX_val_best[0, sel_slice, :, :], cmap = plt.cm.binary)
        plt.subplot(3, 1, 3, title = f'Worst Reconstruction (Parameter #{worst_val_idx})')
        plt.imshow(pX_val_worst[0, sel_slice, :, :], cmap = plt.cm.binary)
        del best_train_idx, worst_train_idx, pX_train, pX_train_gt, pX_train_best, pX_train_worst, pX_val_gt, pX_val_best, pX_val_worst
        return train_figure, val_figure, best_train_loss, worst_train_loss, best_val_loss, worst_val_loss

    ##############################################################################################
    # ------------------------------------- Training Script --------------------------------------
    ##############################################################################################

    # Functionality called upon the Start of Training
    def on_train_start(self):
        
        # Model Training Mode Setup
        self.model.train(); self.skip = 0
        self.automatic_optimization = False
        self.train_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Training Performance')
        self.train_recon = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Training Reconstructions')
        self.train_logger.experiment.add_graph(self.model, (torch.rand(1, 1), 
                                                            torch.rand(1, self.settings.num_labels),
                                                            torch.rand(1, self.settings.num_labels)))

        # Training & Validation Set Example Patient Dataset 
        #self.data = MUDI_1D.load(self.settings.data_folderpath, self.settings.data_version)
        self.data = MUDI_1D(self.settings.data_settings)
        self.pX, self.pMask = self.data.get_patient(self.settings.sel_train_patient)

        # Full Batch Label Data Handling
        self.y_batch = torch.Tensor(self.data.params.iloc[np.hstack((self.data.idxh_train,
                                        self.data.idxh_val)), :].values).type(torch.float)
        assert(self.y_batch.shape[0] == self.data.num_params
               ), f"ERROR: Batch Labels wrongly Set for Reconstruction!"

        # Training & Validation Selected Parameter for 
        assert(self.settings.sel_train_param < self.data.num_train_params), f"ERROR: Selected Target Parameters not Valid!"
        assert(self.settings.sel_val_param < self.data.num_params), f"ERROR: Selected Target Parameters not Valid!"
        self.py_train = self.y_batch[self.settings.sel_train_param, :].repeat(self.pX.shape[0], 1)
        self.py_val = self.y_batch[self.settings.sel_val_param, :].repeat(self.pX.shape[0], 1)

    # Functionality called upon the Start of Training Epoch
    def on_train_epoch_start(self):

        # Loss Value Initialization
        self.num_epochs = self.past_epochs + self.current_epoch
        self.tt_loss = 0; self.tv_loss = 0
        #self.vt_loss = 0; self.vv_loss = 0
        self.train_loss = 0 #; self.val_loss = 0

        # Random Selection of Parameters for Reconstruction
        self.idx_recon_batch = np.hstack((  self.data.idxh_train[np.sort(np.random.choice(self.data.num_train_params,
                                            int((40 * self.data.num_train_params) / 100), replace = False))],
                                            self.data.idxh_val[np.sort(np.random.choice(self.data.num_val_params,
                                            int((40 * self.data.num_val_params) / 100), replace = False))]))

    # --------------------------------------------------------------------------------------------

    # Training Step / Batch Loop 
    def training_step(self, batch, batch_idx):

        # Full Training & Validation Batch Input Data Handling
        X_train_batch, y_train_batch = batch['train']                                   # Training Batch   | [batch_size * train_params, 1] Voxels + [batch_size * train_params, 5] Labels
        X_val_batch, y_val_batch = batch['val']                                         # Validation Batch | [batch_size * val_params, 1] Voxels + [batch_size * val_params, 5] Labels
        X_train_batch = X_train_batch.view(X_train_batch.shape[0],
                                1).type(torch.float).to(self.settings.device)           # Each of the 2 Batches contains a No. (batch_size) of what I refer to here, as a Single Batch
        y_train_batch = y_train_batch.type(torch.float).to(self.settings.device)        # or the Combination of all the Voxels in the same Position for all 1344 Images
        X_val_batch = X_val_batch.type(torch.float).to(self.settings.device)
        y_val_batch = y_val_batch.type(torch.float).to(self.settings.device)
        assert(X_train_batch.shape[0] % self.data.num_train_params == 0
               ), f"ERROR: Batch Labels wrongly Set for Reconstruction!"
        assert(X_val_batch.shape[0] % self.data.num_val_params == 0
               ), f"ERROR: Batch Labels wrongly Set for Reconstruction!"

        # Voxel Reconstruction Loop (all Parameters)
        tt_loss = torch.zeros(1); tv_loss = torch.zeros(1)
        #vt_loss = torch.zeros(1); vv_loss = torch.zeros(1)
        for i in self.idx_recon_batch:                                                  # Parameter Loop   | for i in range(1344):

            # Forward Propagation
            y_target = self.y_batch[i, :].repeat(X_train_batch.shape[0], 1)             # Target Parameter | Repetition of a Single Batch of Labels | [batch_size * 1344, 5] Labels
            X_target = self.model(X_train_batch, y_train_batch, y_target)               # Target Voxels    | Model Application                      | [batch_size * train_params, 1] Voxels
                                                                                        #                  | All the Voxels in the Training Batch are turned into the Current Target Parameter
            # Train -> Train Set Loss Computation
            if i < self.data.num_train_params:
                idx_gt = np.where(np.array(range(X_train_batch.shape[0])) % self.data.num_train_params == i)                # Index of GT Batch Samples, or Samples with the Target Parameters (1 per 'Batch')
                #idx_gt = np.where(y_train_batch == self.data.params.iloc[i, :])                                            # Utilizing the Parameter DataFrame does not work due to Torch Decimals
                X_gt = X_train_batch[idx_gt].repeat(self.data.num_train_params, 1).T.ravel().view(X_target.shape[0], 1)
                tt_loss = tt_loss + (self.criterion(X_target, X_gt) / self.data.num_train_params)
         
            # Train -> Validation Set Loss Computation
            else:
                idx_gt = np.where(np.array(range(X_val_batch.shape[0])) % self.data.num_val_params == (i - self.data.num_train_params))
                #idx_gt = np.where(y_val_batch == self.data.params.iloc[i, :])
                X_gt = X_val_batch[idx_gt].repeat(self.data.num_train_params, 1).T.ravel().view(X_target.shape[0], 1)
                tv_loss = tv_loss + (self.criterion(X_target, X_gt) / self.data.num_train_params)
                                        
        # Backwards Propagation
        self.optimizer.zero_grad()
        train_loss = tt_loss + tv_loss
        self.log('loss', train_loss)
        train_loss.backward()
        #tt_loss.backward(retain_graph = True)
        #tv_loss.backward(retain_graph = True)
        self.optimizer.step()

        del X_train_batch, X_val_batch, y_train_batch, y_val_batch, idx_gt, X_gt, X_target, y_target
        return {"tt_loss": tt_loss, "tv_loss": tv_loss, 'train_loss': tt_loss + tv_loss}

    # Functionality called upon the End of a Batch Training Step
    def on_train_batch_end(self, loss, batch, batch_idx):
        self.tt_loss = self.tt_loss + loss['tt_loss'].item()
        self.tv_loss = self.tv_loss + loss['tv_loss'].item()
        #self.vt_loss = self.vt_loss + loss['vt_loss'].item()
        #self.vv_loss = self.vv_loss + loss['vv_loss'].item()
        self.train_loss = self.train_loss + loss['train_loss'].item()
        #self.val_loss = self.val_loss + loss['val_loss'].item()

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Training Epoch
    def on_train_epoch_end(self):

        # Learning Rate Decay
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # Epoch Update for Losses & Reconstructed Images
        self.num_epochs = self.past_epochs + self.current_epoch
        print(self.num_epochs)
        self.tt_loss = self.tt_loss / self.train_batches
        self.tv_loss = self.tv_loss / self.val_batches
        #self.vt_loss = self.vt_loss / self.train_batches
        #self.vv_loss = self.vv_loss / self.val_batches
        self.train_loss = self.train_loss / self.train_batches
        #self.val_loss = self.val_loss / self.train_batches

        # TensorBoard Logger Model Visualizer, Update for Scalar Values & Image Visualizer
        self.train_logger.experiment.add_scalar("Train -> Train Reconstruction Loss", self.tt_loss, self.num_epochs)
        self.train_logger.experiment.add_scalar("Train -> Val Interpolation Loss", self.tv_loss, self.num_epochs)
        #self.train_logger.experiment.add_scalar("Val -> Train Interpolation Loss", self.vt_loss, self.num_epochs)
        #self.train_logger.experiment.add_scalar("Val -> Val Reconstruction Loss", self.vv_loss, self.num_epochs)
        self.train_logger.experiment.add_scalar("Total Training Loss", self.train_loss, self.num_epochs)
        #self.train_logger.experiment.add_scalar("Total Validation Loss", self.val_loss, self.num_epochs)

        # TensorBoard Logger Example Patient Image Reconstructions
        train_plot, val_plot, recon_train_best, recon_train_worst, recon_val_best, recon_val_worst = self.reconstruct(sel_slice = 25)
        self.train_recon.experiment.add_scalar("Best Training Image Reconstruction Loss", recon_train_best, self.num_epochs)
        self.train_recon.experiment.add_scalar("Worst Training Image Reconstruction Loss", recon_train_worst, self.num_epochs)
        self.train_recon.experiment.add_figure(f'Training Image Reconstruction', train_plot, self.num_epochs)
        self.train_recon.experiment.add_scalar("Best Validation Image Reconstruction Loss", recon_val_best, self.num_epochs)
        self.train_recon.experiment.add_scalar("Worst Validation Image Reconstruction Loss", recon_val_worst, self.num_epochs)
        self.train_recon.experiment.add_figure(f'Validation Image Reconstruction', val_plot, self.num_epochs)

        # Model Checkpoint Saving
        torch.save({'ModelSD': self.model.state_dict(),
                    'OptimizerSD': self.optimizer.state_dict(),
                    'Training Epochs': self.num_epochs,
                    'RNG State': torch.get_rng_state()},
                    self.model_filepath)

"""
    ##############################################################################################
    # -------------------------------------- Testing Script --------------------------------------
    ##############################################################################################

    # Functionality called upon the Start of Testing
    def on_test_start(self):
        
        # Model Testing Mode Setup
        self.model.eval()
        self.automatic_optimization = False
        self.test_logger = TensorBoardLogger(f'{self.settings.save_folderpath}/V{self.settings.model_version}', 'Testing Performance')
        self.test_logger.experiment.add_graph(self.model, ( torch.rand(1), 
                                                            torch.rand(self.settings.num_labels),
                                                            torch.rand(self.settings.num_labels)))
        
        # Training & Validation Set Example Patient Dataset 
        self.data = MUDI_1D.load(self.settings.data_folderpath, self.settings.data_version)
        self.ex_patient = 4; self.pX, self.pMask = self.data.get_patient(self.ex_patient)

    # Functionality called upon the Start of Testing Epoch
    def on_test_epoch_start(self):
        self.num_epochs = self.past_epochs + self.current_epoch
        self.tt_loss = 0; self.vt_loss = 0
        self.tv_loss = 0; self.vv_loss = 0
        self.train_loss = 0; self.val_loss = 0

    # --------------------------------------------------------------------------------------------

    # Testing Step / Batch Loop 
    def test_step(self, batch, batch_idx):

        # Full Batch Input Data Handling
        X_train_batch, y_train_batch = batch['train']
        X_val_batch, y_val_batch = batch['val']
        X_batch = torch.cat((X_train_batch, X_val_batch), dim = 0)
        y_batch = torch.cat((y_train_batch, y_val_batch), dim = 0)
        X_batch = X_batch.type(torch.float).to(self.settings.device)
        y_batch = y_batch.type(torch.float).to(self.settings.device)

        # Voxel Reconstruction Loop (all Parameters)
        assert(y_batch.shape == self.data.params.shape
               ), f"ERROR: Batch Labels wrongly Set for Reconstruction!"
        tt_loss = torch.zeros(1); tv_loss = torch.zeros(1)
        vt_loss = torch.zeros(1); vv_loss = torch.zeros(1)
        for i in range(self.data.num_params):
            
            # Forward Propagation
            y_target = y_batch[i, :].repeat(self.data.num_params, 1)
            X_target = self.model(X_batch, y_batch, y_target)

            # Loss Computation
            t_loss = self.criterion(X_target[0:self.data.train_params, :],
                            X_batch[i, :].repeat(self.data.train_params, 1))
            v_loss = self.criterion(X_target[self.data.train_params::, :],
                            X_batch[i, :].repeat(self.data.val_params, 1))
            if i < self.data.train_params:
                tt_loss = tt_loss + (t_loss / self.data.train_params)
                vt_loss = vt_loss + (v_loss / self.data.val_params)
            else:
                tv_loss = tv_loss + (t_loss / self.data.train_params)
                vv_loss = vv_loss + (v_loss / self.data.val_params)
            del X_target, y_target, t_loss, v_loss
        del X_batch, X_train_batch, X_val_batch, y_train_batch, y_val_batch, y_batch
        return {"tt_loss": tt_loss, "vt_loss": vt_loss, 'train_loss': tt_loss + tv_loss,
                'tv_loss': tv_loss, 'vv_loss': vv_loss, 'val_loss': vt_loss + vv_loss,}

    # Functionality called upon the End of a Batch Testing Step
    def on_test_batch_end(self, loss, batch, batch_idx):
        self.tt_loss = self.tt_loss + loss['tt_loss'].item()
        self.vt_loss = self.vt_loss + loss['vt_loss'].item()
        self.tv_loss = self.tv_loss + loss['tv_loss'].item()
        self.vv_loss = self.vv_loss + loss['vv_loss'].item()
        self.train_loss = self.train_loss + loss['train_loss'].item()
        self.val_loss = self.val_loss + loss['val_loss'].item()

    # --------------------------------------------------------------------------------------------

    # Functionality called upon the End of a Testing Epoch
    def on_test_epoch_end(self):

        # Learning Rate Decay
        if (self.trainer.current_epoch + 1) in self.lr_decay_epochs:
            self.lr_schedule.step()

        # Epoch Update for Losses & Reconstructed Images
        self.tt_loss = self.tt_loss / self.test_batches
        self.vt_loss = self.vt_loss / self.test_batches
        self.tv_loss = self.tv_loss / self.test_batches
        self.vv_loss = self.vv_loss / self.test_batches
        self.train_loss = self.train_loss / self.test_batches
        self.val_loss = self.val_loss / self.test_batches
        test_plot, recon_loss = self.reconstruct(   mode = 'Test',
                                                    sel_params = 0,
                                                    sel_slice = 25)

        # TensorBoard Logger Model Visualizer, Update for Scalar Values & Image Visualizer
        self.test_logger.experiment.add_scalar("Train -> Train Reconstruction Loss", self.tt_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Val -> Train Interpolation Loss", self.vt_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Train -> Val Interpolation Loss", self.tv_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Val -> Val Reconstruction Loss", self.vv_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Total Training Loss", self.train_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Total Validation Loss", self.val_loss, self.num_epochs)
        self.test_logger.experiment.add_scalar("Image Reconstruction Loss", recon_loss, self.num_epochs)
        self.test_logger.experiment.add_figure(f'Image Reconstruction', test_plot, self.num_epochs)
"""