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
from MUDI_CVAE_GAN_2D import MUDI_CVAE_GAN_2D

# CVAE-GAN 2D Model Class Importing
sys.path.append("Model Builds")
from Encoder import Encoder
from Decoder import Decoder
from Discriminator import Discriminator

# CVAE-GAN 2D Callback Class Importing
from EarlyStopping import EarlyStopping
from SSIMLoss import SSIM, SSIM3D, ssim3D, ssim
from ResultCallback import ResultCallback

# --------------------------------------------------------------------------------------------

# 2D CVAE-GAN Model Training Script (V0)
def CVAE_GAN_2D_train(
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
    train_cvae_logger = TensorBoardLogger(checkpoint_folderpath, 'train/cvae')
    val_cvae_logger = TensorBoardLogger(checkpoint_folderpath, 'validation/cvae')
    train_gan_logger = TensorBoardLogger(checkpoint_folderpath, 'train/gan')
    val_gan_logger = TensorBoardLogger(checkpoint_folderpath, 'validation/gan')
    train_results_logger = TensorBoardLogger(checkpoint_folderpath, 'train/results')

    # Training & Validation DataLoaders Initialization
    train_set = []; val_set = []; viz_logger = []
    train_loader = []; val_loader = []
    for p, patient_id in enumerate(settings.patient_list):

        # Patient Set & DataLoader Initialization
        if patient_id in settings.train_patient_list:
            print(f"Patient #{patient_id} | Training Set:")
            train_set.append(   MUDI_CVAE_GAN_2D(   settings, subject = [patient_id], random = True,
                                                    source_param = settings.train_source_param,
                                                    target_param = settings.train_target_param,
                                                    target_slice = settings.train_target_slice,
                                                    param_loop = settings.train_param_loop))
            train_loader.append(DataLoader(         dataset = train_set[-1], pin_memory = True,
                                                    shuffle = settings.train_sample_shuffle,
                                                    num_workers = settings.num_workers,
                                                    batch_size = settings.batch_size))
            
        elif patient_id in settings.val_patient_list:
            print(f"Patient #{patient_id} | Validation Set:")
            val_set.append(     MUDI_CVAE_GAN_2D(   settings, subject = [patient_id], random = False,
                                                    source_param = settings.val_source_param,
                                                    target_param = settings.val_target_param,
                                                    target_slice = settings.val_target_slice,
                                                    param_loop = settings.val_param_loop))
            val_loader.append(  DataLoader(         dataset = val_set[-1], pin_memory = True,
                                                    shuffle = settings.val_sample_shuffle,
                                                    num_workers = settings.num_workers,
                                                    batch_size = val_set[-1].s_target))
            viz_logger.append(  TensorBoardLogger(checkpoint_folderpath, f'validation/p{patient_id}'))
            
        else: print(f"Patient #{patient_id} | Test Set:     > Not Included")
    
    ##############################################################################################
    # -------------------------------- Setup | Models & Optimizers -------------------------------
    ##############################################################################################

    # Model Initialization
    print(f"Running\n     > Training 2D CVAE-GAN Model with {torch.cuda.device_count()} GPUs!")
    enc_model = nn.DataParallel(Encoder(settings), device_ids = settings.device_ids).to(settings.device)
    dec_model = nn.DataParallel(Decoder(settings), device_ids = settings.device_ids).to(settings.device)
    gan_model = nn.DataParallel(Discriminator(settings), device_ids = settings.device_ids).to(settings.device)

    # Optimizer Initialization
    enc_optimizer = torch.optim.AdamW(  enc_model.parameters(), lr = settings.base_lr,
                                        weight_decay = settings.weight_decay)
    dec_optimizer = torch.optim.AdamW(  dec_model.parameters(), lr = settings.base_lr,
                                        weight_decay = settings.weight_decay)
    gan_optimizer = torch.optim.AdamW(  gan_model.parameters(), lr = settings.base_lr,
                                        weight_decay = settings.weight_decay)

    # Learning Rate Scheduling
    gan_equilibrium = settings.base_equilibrium
    gan_margin = settings.base_margin
    dec_lambda = settings.base_lambda
    enc_lr = ExponentialLR(enc_optimizer, gamma = settings.lr_decay)
    dec_lr = ExponentialLR(dec_optimizer, gamma = settings.lr_decay)
    gan_lr = ExponentialLR(gan_optimizer, gamma = settings.lr_decay)

    # Criterion & Early Stopping Setup
    mse_criterion = nn.MSELoss(reduction = 'mean')
    ssim_criterion = SSIM(window_size = settings.kernel_size)
    bce_criterion = nn.BCELoss(reduction = 'mean')
    earlyStopping = EarlyStopping(settings); train_iter = 0; val_iter = 0

    # Model Checkpoint Loading
    model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best 2D CVAE-GAN.pt")
    if model_filepath.exists():
        checkpoint = torch.load(model_filepath, map_location = settings.device)
        enc_model.load_state_dict(checkpoint['Encoder Model']); enc_optimizer.load_state_dict(checkpoint['Encoder Optimizer'])
        dec_model.load_state_dict(checkpoint['Decoder Model']); dec_optimizer.load_state_dict(checkpoint['Decoder Optimizer'])
        gan_model.load_state_dict(checkpoint['Discriminator Model']); gan_optimizer.load_state_dict(checkpoint['Discriminator Optimizer'])
        train_iter = checkpoint['Current Iteration']; val_iter = checkpoint['Current Validation Iteration']
        save_epoch = checkpoint['Current Epoch']; torch.set_rng_state(checkpoint['RNG State'])
        print(f"     > Loading 2D CVAE-GAN Model for {settings.model_version}: {save_epoch} Past Epochs"); del checkpoint
    else: save_epoch = -1
    
    # Epoch Iteration Loop
    for epoch in range(save_epoch + 1, settings.num_epochs):

        ##############################################################################################
        # ----------------------------------- Training | Iteration -----------------------------------
        ##############################################################################################

        # Result Value Initialization
        train_mse = []; best_train_mse = 1000; worst_train_mse = 0
        train_ssim = []; best_train_ssim = 0; worst_train_ssim = 1000
        train_bce = []; best_train_bce = 1000; worst_train_bce = 0
        train_nle = []; train_kld = []

        # Training Patient Loop
        print(f"Training Epoch #{epoch}:")
        for p, patient_id in enumerate(settings.train_patient_list):

            # Training Iteration Loop
            train_bar = tqdm(   enumerate(train_loader[p]), total = len(train_loader[p]),
                desc = f'Epoch #{epoch} | Training Patient {patient_id}', unit = 'Batches')
            for batch_idx, batch in train_bar:
                
                # --------------------------------------------------------------------------------------------

                # Model & Optimizer Setup
                enc_model.train(); enc_model.zero_grad(); enc_optimizer.zero_grad()
                dec_model.train(); dec_model.zero_grad(); dec_optimizer.zero_grad()
                gan_model.train(); gan_model.zero_grad(); gan_optimizer.zero_grad()

                # Forward Propagation
                mu, logvar = enc_model( batch['X_train'].to(settings.device),
                                        batch['y_train'].to(settings.device))               # Encoder Model Application
                z = Decoder.reparam(mu, logvar)                                             # Reparametrization Trick
                X_target = dec_model(z, batch['y_target'].to(settings.device))              # Decoder Model Application
                auth_pred = gan_model(  batch['X_target'].to(settings.device), X_target)    # Discriminator Model Application
                auth_train = auth_pred[:len(auth_pred) // 2]; auth_target = auth_pred[len(auth_pred) // 2:]
                gc.collect(); torch.cuda.empty_cache()

                # --------------------------------------------------------------------------------------------

                # CVAE Model Loss Computation
                mse_loss = mse_criterion(X_target, batch['X_target'].to(settings.device))               # Mean Squared Error Loss
                ssim_loss = ssim_criterion(X_target.detach().cpu(), torch.Tensor(batch['X_target']))    # Structural Similarity Index
                kld_loss = Decoder.kl_loss(mu, logvar); del mu, logvar, z                               # Kullback-Leibler Divergence
                enc_loss = torch.sum(kld_loss) + torch.sum(mse_loss)                                    # Full CVAE Encoder Loss

                # GAN Model Loss Computation
                bce_train = -torch.log(1 - auth_train); bce_target = -torch.log(auth_pred)              # BCE Loss Segments
                print(bce_train)
                bce_loss = torch.sum(bce_train) + torch.sum(bce_target)
                print(bce_loss)
                #bce_loss = bce_criterion(auth_train, auth_target)                                      # Binary Cross Entropy Loss
                nle_loss = torch.mean(- X_target.view((len(X_target), -1)) ** 2 + 0.5 *\
                    (batch['X_target'].to(settings.device).view(len(batch['X_target']), -1)))           # Normal Loss Expectancy Loss
                dec_loss = torch.sum(dec_lambda * mse_loss) - bce_loss                                  # Full CVAE Decoder Loss

                # --------------------------------------------------------------------------------------------
                            
                # Adversarial Training
                gan_train = True; dec_train = True
                if torch.mean(bce_train).data < gan_equilibrium - gan_margin or\
                    torch.mean(bce_target).data < gan_equilibrium - gan_margin: gan_train = False
                if torch.mean(bce_train).data > gan_equilibrium + gan_margin or\
                    torch.mean(bce_target).data > gan_equilibrium + gan_margin: dec_train = False
                if gan_train is False and dec_train is False: gan_train = True; dec_train = True
                del bce_train, bce_target, auth_train, auth_target
                
                # Backward Propagation
                enc_loss.backward(retain_graph = True); enc_optimizer.step()
                if dec_train:
                    dec_loss.backward(retain_graph = True)
                    dec_optimizer.step(); gan_model.zero_grad()
                if gan_train: bce_loss.backward(); gan_optimizer.step()
                gc.collect(); torch.cuda.empty_cache(); del enc_loss, dec_loss

                # --------------------------------------------------------------------------------------------

                # Loss Appending
                train_mse.append(mse_loss.item()); train_ssim.append(ssim_loss); train_kld.append(kld_loss.item())
                train_nle.append(nle_loss.detach().cpu()); train_bce.append(bce_loss.detach().cpu())
                train_cvae_logger.experiment.add_scalar("MSE Loss | Batch", mse_loss.item(), train_iter)
                train_cvae_logger.experiment.add_scalar("SSIM Index | Batch", ssim_loss, train_iter)
                train_cvae_logger.experiment.add_scalar("KL Divergence | Batch", kld_loss.item(), train_iter)
                train_cvae_logger.experiment.add_scalar("Adversarial Training", int(dec_train == True), train_iter)
                train_gan_logger.experiment.add_scalar("BCE Loss | Batch", bce_loss, train_iter)
                train_gan_logger.experiment.add_scalar("NLE Loss | Batch", nle_loss, train_iter)
                train_gan_logger.experiment.add_scalar("Adversarial Training", int(gan_train == True), train_iter)
                train_iter += 1

                # --------------------------------------------------------------------------------------------

                # Result Saving (CVAE MSE Loss)
                if mse_loss.item() < best_train_mse:
                    best_train_mse = mse_loss.item()
                    best_train_mse_info = { 'loss': mse_loss.item(), 'X_pred': X_target.detach().cpu(), 'auth_pred': auth_pred,
                                            'patient_id': patient_id, 'idxv_slice': batch['slice_target'],
                                            'idxh_source': batch['param_source'], 'idxh_target': batch['param_target']}
                if mse_loss.item() > worst_train_mse:
                    worst_train_mse = mse_loss.item()
                    worst_train_mse_info = {'loss': mse_loss.item(), 'X_pred': X_target.detach().cpu(), 'auth_pred': auth_pred,
                                            'patient_id': patient_id, 'idxv_slice': batch['slice_target'],
                                            'idxh_source': batch['param_source'], 'idxh_target': batch['param_target']}
                    
                # Result Saving (CVAE SSIM Index)
                if ssim_loss > best_train_ssim:
                    best_train_ssim = ssim_loss
                    best_train_ssim_info = {'loss': ssim_loss, 'X_pred': X_target.detach().cpu(), 'auth_pred': auth_pred,
                                            'patient_id': patient_id, 'idxv_slice': batch['slice_target'],
                                            'idxh_source': batch['param_source'], 'idxh_target': batch['param_target']}
                if ssim_loss < worst_train_ssim:
                    worst_train_ssim = ssim_loss
                    worst_train_ssim_info = {'loss': ssim_loss, 'X_pred': X_target.detach().cpu(), 'auth_pred': auth_pred,
                                             'patient_id': patient_id, 'idxv_slice': batch['slice_target'],
                                             'idxh_source': batch['param_source'], 'idxh_target': batch['param_target']}

                # Result Saving (CVAE MSE Loss)
                if bce_loss < best_train_bce:
                    best_train_bce = bce_loss
                    best_train_bce_info = {'loss': bce_loss, 'X_pred': X_target.detach().cpu(), 'auth_pred': auth_pred,
                                           'patient_id': patient_id, 'idxv_slice': batch['slice_target'],
                                           'idxh_source': batch['param_source'], 'idxh_target': batch['param_target']}
                if bce_loss > worst_train_bce:
                    worst_train_bce = bce_loss
                    worst_train_bce_info = {'loss': bce_loss, 'X_pred': X_target.detach().cpu(), 'auth_pred': auth_pred,
                                            'patient_id': patient_id, 'idxv_slice': batch['slice_target'],
                                            'idxh_source': batch['param_source'], 'idxh_target': batch['param_target']}
                gc.collect(); torch.cuda.empty_cache()
                del auth_pred, X_target,

        ##############################################################################################
        # ------------------------------------ Training | Results ------------------------------------
        ##############################################################################################

            # Inter-Patient Reconstruction Parameter Sharing Functionality
            if settings.interpatient_sharing:
                if p == 0: train_set[p].shuffle(); idxh_target_train = train_set[p].idxh_target
                else:
                    train_set[p].shuffle(idxh_target = idxh_target_train)
                    assert(np.all(  train_set[p].idxh_target == train_set[0].idxh_target)
                                    ), f"     > ERROR: Parameter Sharing incorrectly setup!"
            else: train_set[p].shuffle()

        # Learning Rate & GAN Settings Update
        enc_lr.step(); dec_lr.step(); gan_lr.step()
        train_gan_logger.experiment.add_scalar("Equilibrium", gan_equilibrium, epoch); gan_equilibrium *= settings.equilibrium_decay
        train_gan_logger.experiment.add_scalar("Margin", gan_margin, epoch); gan_margin *= settings.margin_decay
        train_cvae_logger.experiment.add_scalar("Lambda", dec_lambda, epoch); dec_lambda *= settings.lambda_decay
        if gan_margin > gan_equilibrium: gan_equilibrium = gan_margin
        if dec_lambda > 1: dec_lambda = 1

        # --------------------------------------------------------------------------------------------
        
        # End of Training Epoch Mean & STD Loss Writing
        train_cvae_logger.experiment.add_scalar("MSE Loss | Mean", np.mean(np.array(train_mse)), epoch)
        train_cvae_logger.experiment.add_scalar("MSE Loss | STD", np.std(np.array(train_mse)), epoch)
        train_cvae_logger.experiment.add_scalar("SSIM Index | Mean", np.mean(np.array(train_ssim)), epoch)
        train_cvae_logger.experiment.add_scalar("SSIM Index | STD", np.std(np.array(train_ssim)), epoch)
        train_cvae_logger.experiment.add_scalar("Mean KL Divergence | Mean", np.mean(np.array(train_kld)), epoch)
        train_cvae_logger.experiment.add_scalar("Mean KL Divergence | Mean", np.std(np.array(train_kld)), epoch)
        train_gan_logger.experiment.add_scalar("BCE Loss | Mean", np.mean(np.array(train_bce)), epoch)
        train_gan_logger.experiment.add_scalar("BCE Loss | STD", np.std(np.array(train_bce)), epoch)
        train_gan_logger.experiment.add_scalar("NLE Loss | Mean", np.mean(np.array(train_nle)), epoch)
        train_gan_logger.experiment.add_scalar("NLE Loss | STD", np.std(np.array(train_nle)), epoch)

        # End of Training Epoch Image Result Writing
        train_results_logger = ResultCallback(  settings, logger = train_results_logger, mode = 'Train', epoch = epoch,
                                                criterion = nn.MSELoss(reduction = 'none'), loss = 'MSE Loss',
                                                best_info = best_train_mse_info, worst_info = worst_train_mse_info)
        #train_results_logger = ResultCallback(  settings, logger = train_results_logger, mode = 'Train', epoch = epoch,
        #                                        criterion = ssim_criterion, loss = 'SSIM Index',
        #                                        best_info = best_train_ssim_info, worst_info = worst_train_ssim_info)
        #train_results_logger = ResultCallback(  settings, logger = train_results_logger, mode = 'Train', epoch = epoch,
        #                                        criterion = nn.BCELoss(reduction = 'none'), loss = 'BCE Loss',
        #                                        best_info = best_train_bce_info, worst_info = worst_train_bce_info)

        ##############################################################################################
        # ---------------------------------- Validation | Iteration ----------------------------------
        ##############################################################################################