# Library Imports
import os
import gc
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import tqdm
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from nilearn.masking import unmask
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_fcgCVAE import MUDI_fcgCVAE

# CVAE-GAN 2D Model Class Importing
sys.path.append("Model Builds")
from Encoder import Encoder
from Decoder import Decoder
from fcgCVAE import fcgCVAE

# CVAE-GAN 2D Callback Class Importing
from EarlyStopping import EarlyStopping
#from SSIMLoss import SSIM, SSIM3D, ssim3D, ssim
from ResultCallback import ResultCallback

# --------------------------------------------------------------------------------------------

# 2D fcgCVAE Model Training Script (V0)
def fcgCVAE_2D_train(
    settings,
):

    ##############################################################################################
    # ------------------------------------ Setup | DataLoader ------------------------------------
    ##############################################################################################

    # Seed Random State for Reproducibility
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    # Experiment Logs Directories Initialization
    checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs')
    train_mse_logger = TensorBoardLogger(checkpoint_folderpath, 'train/mse')
    train_kld_logger = TensorBoardLogger(checkpoint_folderpath, 'train/kld')
    train_ssim_logger = TensorBoardLogger(checkpoint_folderpath, 'train/ssim')
    val_mse_logger = TensorBoardLogger(checkpoint_folderpath, 'val/mse')
    val_kld_logger = TensorBoardLogger(checkpoint_folderpath, 'val/kld')
    val_ssim_logger = TensorBoardLogger(checkpoint_folderpath, 'val/ssim')

    # Training & Validation DataLoaders Initialization
    train_set = []; val_set = []; train_loader = []; val_loader = []
    for p, patient_id in enumerate(settings.patient_list):

        # Patient Set & DataLoader Initialization
        if patient_id in settings.train_patient_list:
            print(f"Patient #{patient_id} | Training Set:")
            train_set.append(   MUDI_fcgCVAE(   settings, subject = [patient_id], random = False,
                                                target_param = settings.train_target_param,
                                                target_slice = settings.train_target_slice,
                                                target_combo = settings.train_param_loop))
            train_loader.append(DataLoader(     dataset = train_set[-1], pin_memory = True,
                                                shuffle = settings.train_sample_shuffle,
                                                num_workers = settings.num_workers,
                                                batch_size = settings.batch_size))
            
        elif patient_id in settings.val_patient_list:
            print(f"Patient #{patient_id} | Validation Set:")
            val_set.append(     MUDI_fcgCVAE(   settings, subject = [patient_id], random = True,
                                                target_param = settings.val_target_param,
                                                target_slice = settings.val_target_slice,
                                                target_combo = settings.val_param_loop))
            val_loader.append(  DataLoader(     dataset = val_set[-1], pin_memory = True,
                                                shuffle = settings.val_sample_shuffle,
                                                num_workers = settings.num_workers, batch_size = 10))
                                                #batch_size = settings.batch_size))
            
        else: print(f"Patient #{patient_id} | Test Set:     > Not Included")
    
    ##############################################################################################
    # -------------------------------- Setup | Models & Optimizers -------------------------------
    ##############################################################################################

    # Model & Optimizer Initialization
    print(f"Running\n     > Training 2D fcgCVAE Model with {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(fcgCVAE(settings), device_ids = settings.device_ids).to(settings.device)
    optimizer = torch.optim.AdamW(  model.parameters(), lr = settings.base_lr,
                                    weight_decay = settings.weight_decay)

    # Criterion & Early Stopping Setup
    mse_criterion = nn.MSELoss(reduction = 'mean')
    #ssim_criterion = SSIM(window_size = settings.kernel_size)
    lr_schedule = ExponentialLR(optimizer, gamma = settings.lr_decay)
    earlyStopping = EarlyStopping(settings); train_iter = 0; val_iter = 0

    # Model Checkpoint Loading
    model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best 2D fcgCVAE.pt")
    if model_filepath.exists():
        checkpoint = torch.load(model_filepath, map_location = settings.device)
        model.load_state_dict(checkpoint['Model'])
        optimizer.load_state_dict(checkpoint['Optimizer'])
        train_iter = checkpoint['Current Training Iteration']
        val_iter = checkpoint['Current Validation Iteration']
        save_epoch = checkpoint['Current Epoch']
        torch.set_rng_state(checkpoint['RNG State']); del checkpoint
        print(f"     > Loading 2D fcgCVAE Model for {settings.model_version}: {save_epoch} Past Epochs")
    else: save_epoch = -1
    
    # Epoch Iteration Loop
    for epoch in range(save_epoch + 1, settings.num_epochs):

        ##############################################################################################
        # ----------------------------------- Training | Iteration -----------------------------------
        ##############################################################################################

        # Result Value Initialization
        train_mse = []; train_kld = []; train_ssim = []
        best_train_mse = 1000; worst_train_mse = 0
        best_train_ssim = 0; worst_train_ssim = 1000

        # Training Patient Loop
        print(f"Training Epoch #{epoch}:")
        for p, patient_id in enumerate(settings.train_patient_list):

            # Training Iteration Loop
            train_bar = tqdm(   enumerate(train_loader[p]), total = len(train_loader[p]),
                desc = f'Epoch #{epoch} | Training Patient {patient_id}', unit = 'Batches')
            for batch_idx, batch in train_bar:
                
                # --------------------------------------------------------------------------------------------

                # Forward Propagation
                model.train(); model.zero_grad(); optimizer.zero_grad()
                X_target, kld_loss = model( batch['X_train'].to(settings.device),                       # Model Application
                                            batch['y_target'].to(settings.device))                      # Kullback-Leibler Divergence
                gc.collect(); torch.cuda.empty_cache()

                # Loss Computation
                mse_loss = mse_criterion(X_target, batch['X_target'].to(settings.device))               # Mean Squared Error Loss
                #ssim_loss = ssim_criterion(X_target.detach().cpu(), torch.Tensor(batch['X_target']))    # Structural Similarity Index
                ssim_loss = np.empty(X_target.shape[0])                                                 # Structural Similarity Index
                for i in range(X_target.shape[0]):
                    ssim_loss[i], ssim_img = ssim( X_target[i, 0, :, :].detach().cpu().numpy().astype(np.float32),
                                                batch['X_target'][i, 0, :, :].detach().cpu().numpy().astype(np.float32), full = True,
                        data_range = (torch.max(batch['X_target'][i, 0, :, :]) - torch.min(batch['X_target'][i, 0, :, :])).cpu().numpy())
                ssim_loss = np.mean(ssim_loss); del ssim_img
                loss = torch.autograd.Variable(1 - torch.from_numpy(np.array([ssim_loss])), requires_grad = True)

                # Backward Propagation
                mse_loss.backward(retain_graph = True)
                kld_loss.backward(retain_graph = True)
                loss.backward(); optimizer.step()
                train_mse_logger.experiment.add_scalar("Batch Loss", mse_loss.item(), train_iter)
                train_kld_logger.experiment.add_scalar("Batch Loss", kld_loss.item(), train_iter)
                train_ssim_logger.experiment.add_scalar("Batch Loss", ssim_loss, train_iter); train_iter += 1
                train_mse.append(mse_loss.item()); train_kld.append(kld_loss.item())
                train_ssim.append(ssim_loss); gc.collect(); torch.cuda.empty_cache()

                # --------------------------------------------------------------------------------------------

                # Result Saving (MSE Loss)
                if mse_loss.item() < best_train_mse:
                    best_train_mse = mse_loss.item()
                    best_train_mse_info = { 'loss': mse_loss.item(), 'patient_id': patient_id,
                                            'X_gt': batch['X_target'].detach().cpu(),
                                            'X_fake': X_target.detach().cpu(),
                                            'idxv_slice': batch['slice_target'],
                                            'idxh_target': batch['param_target']}
                if mse_loss.item() > worst_train_mse:
                    worst_train_mse = mse_loss.item()
                    worst_train_mse_info = {'loss': mse_loss.item(), 'patient_id': patient_id,
                                            'X_gt': batch['X_target'].detach().cpu(),
                                            'X_fake': X_target.detach().cpu(),
                                            'idxv_slice': batch['slice_target'],
                                            'idxh_target': batch['param_target']}
                
                # Result Saving (SSIM Index)
                if ssim_loss > best_train_ssim:
                    best_train_ssim = ssim_loss
                    best_train_ssim_info = {'loss': ssim_loss, 'patient_id': patient_id,
                                            'X_gt': batch['X_target'].detach().cpu(),
                                            'X_fake': X_target.detach().cpu(),
                                            'idxv_slice': batch['slice_target'],
                                            'idxh_target': batch['param_target']}
                if ssim_loss < worst_train_ssim:
                    worst_train_ssim = ssim_loss
                    worst_train_ssim_info = {'loss': ssim_loss, 'patient_id': patient_id,
                                            'X_gt': batch['X_target'].detach().cpu(),
                                            'X_fake': X_target.detach().cpu(),
                                            'idxv_slice': batch['slice_target'],
                                            'idxh_target': batch['param_target']}
                gc.collect(); torch.cuda.empty_cache()
                del batch, X_target, mse_loss, kld_loss, ssim_loss

        ##############################################################################################
        # ------------------------------------ Training | Results ------------------------------------
        ##############################################################################################

            # Inter-Patient Reconstruction Parameter Sharing Functionality
            if settings.interpatient_sharing:
                if p == 0:
                    train_set[p].shuffle()
                    train_idxh_target = train_set[p].idxh_target
                    train_idxv_slice = train_set[p].idxv_slice
                    train_idx_combo = train_set[p].idx_combo
                else:
                    train_set[p].shuffle(idxh_target = train_idxh_target)
                    train_set[p].shuffle(idxv_slice = train_idxv_slice)
                    train_set[p].shuffle(idx_combo = train_idx_combo)
            else: train_set[p].shuffle()

        # --------------------------------------------------------------------------------------------
        
        # End of Training Epoch Mean & STD Loss Writing
        train_kld_logger.experiment.add_scalar("Learning Rate", lr_schedule.get_last_lr()[0], epoch); lr_schedule.step()
        train_mse_logger.experiment.add_scalar("Mean Loss", np.mean(np.array(train_mse)), epoch)
        train_mse_logger.experiment.add_scalar("Loss STD", np.std(np.array(train_mse)), epoch)
        train_kld_logger.experiment.add_scalar("Mean Loss", np.mean(np.array(train_kld)), epoch)
        train_kld_logger.experiment.add_scalar("Loss STD", np.std(np.array(train_kld)), epoch)
        train_ssim_logger.experiment.add_scalar("Mean Loss", np.mean(np.array(train_ssim)), epoch)
        train_ssim_logger.experiment.add_scalar("Loss STD", np.std(np.array(train_ssim)), epoch)
        del train_mse, train_kld, train_ssim

        # End of Training Epoch Image Result Writing
        train_mse_logger = ResultCallback(  settings, mode = 'Train',
                                            logger = train_mse_logger,
                                            epoch = epoch, loss = 'MSE Loss',
                                            best_info = best_train_mse_info,
                                            worst_info = worst_train_mse_info)
        train_ssim_logger = ResultCallback( settings, mode = 'Train',
                                            logger = train_ssim_logger,
                                            epoch = epoch, loss = 'SSIM Index',
                                            best_info = best_train_ssim_info,
                                            worst_info = worst_train_ssim_info)
        del best_train_mse_info, worst_train_mse_info, best_train_ssim_info, worst_train_ssim_info

        ##############################################################################################
        # ---------------------------------- Validation | Iteration ----------------------------------
        ##############################################################################################

        # Result Value Initialization
        val_mse = []; val_kld = []; val_ssim = []
        best_val_mse = 1000; worst_val_mse = 0
        best_val_ssim = 0; worst_val_ssim = 1000

        # Validation Patient Loop
        with torch.no_grad():
            model.eval(); print(f"Validation Epoch #{epoch}:")
            for p, patient_id in enumerate(settings.val_patient_list):

                # Validation Iteration Loop
                val_bar = tqdm(   enumerate(val_loader[p]), total = len(val_loader[p]),
                    desc = f'Epoch #{epoch} | Validation Patient {patient_id}', unit = 'Batches')
                for batch_idx, batch in val_bar:
                    
                    # --------------------------------------------------------------------------------------------

                    # Forward Propagation
                    X_target, kld_loss = model( batch['X_train'].to(settings.device),                       # Model Application
                                                batch['y_target'].to(settings.device))                      # Kullback-Leibler Divergence
                    gc.collect(); torch.cuda.empty_cache()
                    
                    # Loss Computation
                    mse_loss = mse_criterion(X_target, batch['X_target'].to(settings.device))               # Mean Squared Error Loss
                    ssim_loss = np.empty(X_target.shape[0])                                                 # Structural Similarity Index
                    for i in range(X_target.shape[0]):
                        ssim_loss[i], ssim_img = ssim( X_target[i, 0, :, :].detach().cpu().numpy().astype(np.float32),
                                                    batch['X_target'][i, 0, :, :].detach().cpu().numpy().astype(np.float32), full = True,
                            data_range = (torch.max(batch['X_target'][i, 0, :, :]) - torch.min(batch['X_target'][i, 0, :, :])).cpu().numpy())
                    ssim_loss = np.mean(ssim_loss); del ssim_img
                    
                    # Loss Appending
                    val_mse_logger.experiment.add_scalar("Batch Loss", mse_loss.item(), val_iter)
                    val_kld_logger.experiment.add_scalar("Batch Loss", kld_loss.item(), val_iter)
                    val_ssim_logger.experiment.add_scalar("Batch Loss", ssim_loss, val_iter); val_iter += 1
                    val_mse.append(mse_loss.item()); val_kld.append(kld_loss.item())
                    val_ssim.append(ssim_loss); gc.collect(); torch.cuda.empty_cache()

                    # --------------------------------------------------------------------------------------------

                    # Result Saving (MSE Loss)
                    if mse_loss.item() < best_val_mse:
                        best_val_mse = mse_loss.item()
                        best_val_mse_info = {   'loss': mse_loss.item(), 'patient_id': patient_id,
                                                'X_gt': batch['X_target'].detach().cpu(),
                                                'X_fake': X_target.detach().cpu(),
                                                'idxv_slice': batch['slice_target'],
                                                'idxh_target': batch['param_target']}
                    if mse_loss.item() > worst_val_mse:
                        worst_val_mse = mse_loss.item()
                        worst_val_mse_info = {  'loss': mse_loss.item(), 'patient_id': patient_id,
                                                'X_gt': batch['X_target'].detach().cpu(),
                                                'X_fake': X_target.detach().cpu(),
                                                'idxv_slice': batch['slice_target'],
                                                'idxh_target': batch['param_target']}
                    
                    # Result Saving (SSIM Index)
                    if ssim_loss > best_val_ssim:
                        best_val_ssim = ssim_loss
                        best_val_ssim_info = {  'loss': ssim_loss, 'patient_id': patient_id,
                                                'X_gt': batch['X_target'].detach().cpu(),
                                                'X_fake': X_target.detach().cpu(),
                                                'idxv_slice': batch['slice_target'],
                                                'idxh_target': batch['param_target']}
                    if ssim_loss < worst_val_ssim:
                        worst_val_ssim = ssim_loss
                        worst_val_ssim_info = { 'loss': ssim_loss, 'patient_id': patient_id,
                                                'X_gt': batch['X_target'].detach().cpu(),
                                                'X_fake': X_target.detach().cpu(),
                                                'idxv_slice': batch['slice_target'],
                                                'idxh_target': batch['param_target']}
                    gc.collect(); torch.cuda.empty_cache()
                    del batch, X_target, mse_loss, kld_loss, ssim_loss

            ##############################################################################################
            # ------------------------------------ Validation | Results ------------------------------------
            ##############################################################################################

                # Inter-Patient Reconstruction Parameter Sharing Functionality
                if settings.interpatient_sharing:
                    if p == 0:
                        val_set[p].shuffle()
                        val_idxh_target = val_set[p].idxh_target
                        val_idxv_slice = val_set[p].idxv_slice
                        val_idx_combo = val_set[p].idx_combo
                    else:
                        val_set[p].shuffle(idxh_target = val_idxh_target)
                        val_set[p].shuffle(idxv_slice = val_idxv_slice)
                        val_set[p].shuffle(idx_combo = val_idx_combo)
                else: val_set[p].shuffle()

            # --------------------------------------------------------------------------------------------
            
        # End of Training Epoch Mean & STD Loss Writing
        val_mse_logger.experiment.add_scalar("Mean Loss", np.mean(np.array(val_mse)), epoch)
        val_mse_logger.experiment.add_scalar("Loss STD", np.std(np.array(val_mse)), epoch)
        val_kld_logger.experiment.add_scalar("Mean Loss", np.mean(np.array(val_kld)), epoch)
        val_kld_logger.experiment.add_scalar("Loss STD", np.std(np.array(val_kld)), epoch)
        val_ssim_logger.experiment.add_scalar("Mean Loss", np.mean(np.array(val_ssim)), epoch)
        val_ssim_logger.experiment.add_scalar("Loss STD", np.std(np.array(val_ssim)), epoch)
        del val_kld, val_ssim

        # End of Training Epoch Image Result Writing
        val_mse_logger = ResultCallback(    settings, mode = 'Validation',
                                            logger = val_mse_logger,
                                            epoch = epoch, loss = 'MSE Loss',
                                            best_info = best_val_mse_info,
                                            worst_info = worst_val_mse_info)
        val_ssim_logger = ResultCallback(   settings, mode = 'Validation',
                                            logger = val_ssim_logger,
                                            epoch = epoch, loss = 'SSIM Index',
                                            best_info = best_val_ssim_info,
                                            worst_info = worst_val_ssim_info)
        del best_val_mse_info, worst_val_mse_info, best_val_ssim_info, worst_val_ssim_info

        # Early Stopping Callback Application
        early_stop = earlyStopping( loss = np.mean(np.array(val_mse)), epoch = epoch,
                                    model = model, optimizer = optimizer)
        val_kld_logger.experiment.add_scalar("Early Stopping Counter", earlyStopping.counter, epoch)
        if early_stop: print(f'     > Training Finished at Epoch #{epoch}'); return
            
