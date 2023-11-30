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
from MUDI_probUNet import MUDI_probUNet

# CVAE-GAN 2D Model Class Importing
sys.path.append("Model Builds")
from model import ProbabilisticUnet
from utils import l2_regularisation

# CVAE-GAN 2D Callback Class Importing
from EarlyStopping import EarlyStopping
#from SSIMLoss import SSIM, SSIM3D, ssim3D, ssim
from ResultCallback import ResultCallback

# --------------------------------------------------------------------------------------------

# 2D fcgCVAE Model Training Script (V0)
def probUNet_2D_test(
    settings,
):

    ##############################################################################################
    # -------------------------------- Setup | Models & Optimizers -------------------------------
    ##############################################################################################

    # Seed Random State for Reproducibility
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    # Model & Optimizer Initialization
    print(f"Running\n     > Testing 2D fcgCVAE Model with {torch.cuda.device_count()} GPUs!")
    mse_criterion = nn.MSELoss(reduction = 'mean'); mae_criterion = nn.L1Loss(reduction = 'mean')
    model = ProbabilisticUnet(  input_channels = settings.in_channels + settings.num_labels,
                                num_classes = 1, label_channels = 1, no_convs_fcomb = 4,
                                num_filters = [32, 64, 128, 256], latent_dim = settings.dim_latent,
                                beta = 1.0, norm = True).to(settings.device)
    #optimizer = torch.optim.AdamW(  model.parameters(), lr = settings.base_lr, weight_decay = settings.weight_decay)

    # Model Checkpoint Loading
    model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best 2D fcgCVAE.pt")
    if model_filepath.exists():
        checkpoint = torch.load(model_filepath)#, map_location = settings.device)
        model.load_state_dict(checkpoint['Model'])
        #optimizer.load_state_dict(checkpoint['Optimizer'])
        save_epoch = checkpoint['Current Epoch']
        torch.set_rng_state(checkpoint['RNG State'])
        del checkpoint
        print(f"     > Loading 2D fcgCVAE Model for {settings.model_version}: {save_epoch} Past Epochs")
    else: save_epoch = -1

    ##############################################################################################
    # ------------------------------------ Setup | DataLoader ------------------------------------
    #############################################################################################

    # Testing DataLoaders Initialization
    checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs/test')
    print(f"Patient #{settings.test_patient_list[0]} | Test Set:")
    test_set = MUDI_probUNet(   settings, random = False,
                                subject = [settings.test_patient_list[0]],
                                target_param = 100,#settings.val_target_param,
                                target_slice = 100,#settings.val_target_slice,
                                target_combo = 0)#settings.val_param_loop)
    test_loader = DataLoader(   dataset = test_set, pin_memory = True,
                                shuffle = False,#settings.train_sample_shuffle,
                                num_workers = settings.num_workers,
                                batch_size = test_set.s_target)#settings.batch_size)
    test_logger = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}')
    mse_logger = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/mse')
    mse_logger0 = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/mse0')
    mae_logger = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/mae')
    mae_logger0 = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/mae0')
    ssim_logger = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/ssim')
    ssim_logger0 = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/ssim0')
    
    ##############################################################################################
    # ------------------------------------- Test | Iteration -------------------------------------
    ##############################################################################################

    # Test Patient Loop
    with torch.no_grad():
        model.eval(); test_iter = 0; print(f"Test Epoch:")

        # Testing Iteration Loop
        test_bar = tqdm(   enumerate(test_loader), total = len(test_loader),
            desc = f'Test Patient {settings.test_patient_list[0]}', unit = 'Batches')
        for batch_idx, batch in test_bar:
            
            # --------------------------------------------------------------------------------------------

            # Forward Propagation
            model.forward(  batch['input'].to(settings.device),
                            None, training = False)
            X_target, _ = model.sample(testing = True)
            X_target = X_target.squeeze(dim = 1)
            gc.collect(); torch.cuda.empty_cache()

            # Slice Iteration Loop
            test_mse = []; test_mae = []; test_ssim = []
            for i in range(X_target.shape[0]):
                
                # Batch Loss Computation
                mse_loss = mse_criterion(   X_target[i, :, :].to(settings.device),
                                            batch['X_target'][i, 0, :, :].to(settings.device))
                mae_loss = mae_criterion(   X_target[i, :, :].to(settings.device),
                                            batch['X_target'][i, 0, :, :].to(settings.device))
                ssim_loss, ssim_img = ssim( X_target[i, :, :].detach().cpu().numpy().astype(np.float32),
                                            batch['X_target'][i, 0, :, :].detach().cpu().numpy().astype(np.float32), full = True,
                    data_range = (torch.max(batch['X_target'][i, 0, :, :]) - torch.min(batch['X_target'][i, 0, :, :])).cpu().numpy())

                # Batch Image Results
                plot = plt.figure(figsize = (20, 12)); plt.suptitle(f"Batch #{test_iter} Image Results")
                plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
                plt.subplot(1, 2, 1, title =    f"Original | Target Parameter #{batch['param_target'][i]} | " +\
                                                f"Target Slice #{batch['slice_target'][i]}")
                plt.imshow(batch['X_target'][i, 0, :, :], cmap = plt.cm.binary)
                plt.subplot(1, 2, 2, title =    f"Reconstruction | MSE: {np.round(mse_loss.item(), 5)} | " +\
                                                f"MAE: {np.round(mae_loss.item(), 5)} | SSIM: {np.round(np.mean(ssim_loss), 5)}")
                plt.imshow(X_target[i, :, :].cpu(), cmap = plt.cm.binary)

                # Batch Loss Saving
                test_mse.append(mse_loss.item()); test_mae.append(mae_loss.item()); test_ssim.append(np.mean(ssim_loss)); del ssim_img
                test_logger.experiment.add_figure(f"Batch Image Results", plot, test_iter)
                mse_logger.experiment.add_scalar("Batch Loss", mse_loss.item(), test_iter)
                mae_logger.experiment.add_scalar("Batch Loss", mae_loss.item(), test_iter)
                ssim_logger.experiment.add_scalar("Batch Loss", np.mean(ssim_loss), test_iter); test_iter += 1

            # Best & Worst Result Saving
            best_mse_idx = np.argmin(test_mse); worst_mse_idx = np.argmax(test_mse)
            best_ssim_idx = np.argmax(test_ssim); worst_ssim_idx = np.argmin(test_ssim)
            best_test_mse_info = {  'loss': test_mse[best_mse_idx], 'patient_id': settings.test_patient_list[0],
                                    'X_gt': batch['X_target'][best_mse_idx, 0, :, :].detach().cpu(),
                                    'X_fake': X_target[best_mse_idx, :, :].detach().cpu(),
                                    'idxv_slice': batch['slice_target'][best_mse_idx],
                                    'idxh_target': batch['param_target'][best_mse_idx]}
            worst_test_mse_info = { 'loss': test_mse[worst_mse_idx], 'patient_id': settings.test_patient_list[0],
                                    'X_gt': batch['X_target'][worst_mse_idx, 0, :, :].detach().cpu(),
                                    'X_fake': X_target[worst_mse_idx, :, :].detach().cpu(),
                                    'idxv_slice': batch['slice_target'][worst_mse_idx],
                                    'idxh_target': batch['param_target'][worst_mse_idx]}
            best_test_ssim_info = { 'loss': test_ssim[best_ssim_idx], 'patient_id': settings.test_patient_list[0],
                                    'X_gt': batch['X_target'][best_ssim_idx, 0, :, :].detach().cpu(),
                                    'X_fake': X_target[best_ssim_idx, :, :].detach().cpu(),
                                    'idxv_slice': batch['slice_target'][best_ssim_idx],
                                    'idxh_target': batch['param_target'][best_ssim_idx]}
            worst_test_ssim_info = {'loss': test_ssim[worst_ssim_idx], 'patient_id': settings.test_patient_list[0],
                                    'X_gt': batch['X_target'][worst_ssim_idx, 0, :, :].detach().cpu(),
                                    'X_fake': X_target[worst_ssim_idx, :, :].detach().cpu(),
                                    'idxv_slice': batch['slice_target'][worst_ssim_idx],
                                    'idxh_target': batch['param_target'][worst_ssim_idx]}
            gc.collect(); torch.cuda.empty_cache()

            # End of Testing Epoch Image Result Writing
            if batch_idx < len(test_loader) // 2:
                mse_logger.experiment.add_scalar("Mean Loss", np.mean(test_mse), batch_idx)
                mae_logger.experiment.add_scalar("Mean Loss", np.mean(test_mae), batch_idx)
                ssim_logger.experiment.add_scalar("Mean Loss", np.mean(test_ssim), batch_idx)
                mse_logger = ResultCallback(    settings, mode = 'Test',
                                                logger = mse_logger,
                                                epoch = batch_idx, loss = 'MSE Loss',
                                                best_info = best_test_mse_info,
                                                worst_info = worst_test_mse_info)
                ssim_logger = ResultCallback(   settings, mode = 'Test',
                                                logger = ssim_logger,
                                                epoch = batch_idx, loss = 'SSIM Index',
                                                best_info = best_test_ssim_info,
                                                worst_info = worst_test_ssim_info)
            else:
                mse_logger0.experiment.add_scalar("Mean Loss", np.mean(test_mse), batch_idx)
                mae_logger0.experiment.add_scalar("Mean Loss", np.mean(test_mae), batch_idx)
                ssim_logger0.experiment.add_scalar("Mean Loss", np.mean(test_ssim), batch_idx)
                mse_logger0 = ResultCallback(   settings, mode = 'Test',
                                                logger = mse_logger0,
                                                epoch = batch_idx, loss = 'MSE Loss',
                                                best_info = best_test_mse_info,
                                                worst_info = worst_test_mse_info)
                ssim_logger0 = ResultCallback(  settings, mode = 'Test',
                                                logger = ssim_logger0,
                                                epoch = batch_idx, loss = 'SSIM Index',
                                                best_info = best_test_ssim_info,
                                                worst_info = worst_test_ssim_info)
            #import tensorboard