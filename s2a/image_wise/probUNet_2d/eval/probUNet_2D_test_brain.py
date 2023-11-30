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
import nibabel as nib
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
def probUNet_2D_test_brain(
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
    wm_logger = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/wm')
    gm_logger = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/gm')
    csf_logger = TensorBoardLogger(checkpoint_folderpath, f'epoch{save_epoch}/p{settings.test_patient_list[0]}/csf')

    # Mask Initialization
    wm_mask = MUDI_probUNet.zero_padding(np.array(nib.load(os.path.join(   settings.mask_folderpath,
                                        f'p{settings.test_patient_list[0]}_wm.nii.gz')).dataobj).T.astype(int), settings.img_shape)[0]
    gm_mask = MUDI_probUNet.zero_padding(np.expand_dims(np.array(nib.load(os.path.join(   settings.mask_folderpath,
                                        f'p{settings.test_patient_list[0]}_gm.nii.gz')).dataobj).T.astype(int), 0), settings.img_shape)[0]
    csf_mask = MUDI_probUNet.zero_padding(np.expand_dims(np.array(nib.load(os.path.join(  settings.mask_folderpath,
                                        f'p{settings.test_patient_list[0]}_csf.nii.gz')).dataobj).T.astype(int), 0), settings.img_shape)[0]
    
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

            # Mask Application
            wm_real = torch.Tensor(np.multiply(batch['X_target'][:, 0, :, :], wm_mask))
            wm_gen = torch.Tensor(np.multiply(X_target.detach().cpu(), wm_mask))
            gm_real = torch.Tensor(np.multiply(batch['X_target'][:, 0, :, :], gm_mask))
            gm_gen = torch.Tensor(np.multiply(X_target.detach().cpu(), gm_mask))
            csf_real = torch.Tensor(np.multiply(batch['X_target'][:, 0, :, :], csf_mask))
            csf_gen = torch.Tensor(np.multiply(X_target.detach().cpu(), csf_mask))

            # --------------------------------------------------------------------------------------------

            # White Matter Batch Loss Computation
            sel_slice = random.randint(0, wm_mask.shape[0] - 1)
            wm_mse = mse_criterion(   wm_gen.to(settings.device), wm_real.to(settings.device))
            wm_mae = mae_criterion(   wm_gen.to(settings.device), wm_real.to(settings.device))
            wm_ssim, ssim_img = ssim( wm_gen.detach().cpu().numpy().astype(np.float32),
                                        wm_real.detach().cpu().numpy().astype(np.float32), full = True,
                        data_range = (torch.max(wm_real) - torch.min(wm_real)).cpu().numpy()); del ssim_img

            # White Matter Mask Application
            wm_plot = plt.figure(figsize = (30, 25)); plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
            plt.suptitle(f'Test Patient #{settings.test_patient_list[0]} | White Matter Mask Plotting | Slice #{sel_slice}' +\
                            f'MSE Loss: {wm_mse.item()} | MAE Loss: {wm_mae.item()} | SSIM Index: {np.mean(wm_ssim)}')
            plt.subplot(2, 2, 1, title = 'Original Scan'); plt.imshow(batch['X_target'][sel_slice, 0, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 2, title = 'Generated Scan'); plt.imshow(X_target[sel_slice, :, :].detach().cpu(), cmap = plt.cm.binary)
            plt.subplot(2, 2, 3, title = 'Original Mask'); plt.imshow(wm_real[sel_slice, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 4, title = 'Generated Mask'); plt.imshow(wm_gen[sel_slice, :, :], cmap = plt.cm.binary)
            wm_logger.experiment.add_figure(f"Image Results", wm_plot, test_iter)

            # --------------------------------------------------------------------------------------------

            # Grey Matter Batch Loss Computation
            #sel_slice = random.randint(0, gm_mask.shape[0] - 1)
            gm_mse = mse_criterion(   gm_gen.to(settings.device), gm_real.to(settings.device))
            gm_mae = mae_criterion(   gm_gen.to(settings.device), gm_real.to(settings.device))
            gm_ssim, ssim_img = ssim( gm_gen.detach().cpu().numpy().astype(np.float32),
                                        gm_real.detach().cpu().numpy().astype(np.float32), full = True,
                        data_range = (torch.max(gm_real) - torch.min(gm_real)).cpu().numpy()); del ssim_img

            # White Matter Mask Application
            gm_plot = plt.figure(figsize = (30, 25)); plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
            plt.suptitle(f'Test Patient #{settings.test_patient_list[0]} | Grey Matter Mask Plotting | Slice #{sel_slice}' +\
                            f'MSE Loss: {gm_mse.item()} | MAE Loss: {gm_mae.item()} | SSIM Index: {np.mean(gm_ssim)}')
            plt.subplot(2, 2, 1, title = 'Original Scan'); plt.imshow(batch['X_target'][sel_slice, 0, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 2, title = 'Generated Scan'); plt.imshow(X_target[sel_slice, :, :].detach().cpu(), cmap = plt.cm.binary)
            plt.subplot(2, 2, 3, title = 'Original Mask'); plt.imshow(gm_real[sel_slice, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 4, title = 'Generated Mask'); plt.imshow(gm_gen[sel_slice, :, :], cmap = plt.cm.binary)
            gm_logger.experiment.add_figure(f"Image Results", gm_plot, test_iter)

            # --------------------------------------------------------------------------------------------

            # CSF Batch Loss Computation
            #sel_slice = random.randint(0, csf_mask.shape[0] - 1)
            csf_mse = mse_criterion(   csf_gen.to(settings.device), csf_real.to(settings.device))
            csf_mae = mae_criterion(   csf_gen.to(settings.device), csf_real.to(settings.device))
            csf_ssim, ssim_img = ssim( csf_gen.detach().cpu().numpy().astype(np.float32),
                                        csf_real.detach().cpu().numpy().astype(np.float32), full = True,
                        data_range = (torch.max(csf_real) - torch.min(csf_real)).cpu().numpy()); del ssim_img

            # White Matter Mask Application
            csf_plot = plt.figure(figsize = (30, 25)); plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
            plt.suptitle(f'Test Patient #{settings.test_patient_list[0]} | CSF Mask Plotting | Slice #{sel_slice}' +\
                            f'MSE Loss: {csf_mse.item()} | MAE Loss: {csf_mae.item()} | SSIM Index: {np.mean(csf_ssim)}')
            plt.subplot(2, 2, 1, title = 'Original Scan'); plt.imshow(batch['X_target'][sel_slice, 0, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 2, title = 'Generated Scan'); plt.imshow(X_target[sel_slice, :, :].detach().cpu(), cmap = plt.cm.binary)
            plt.subplot(2, 2, 3, title = 'Original Mask'); plt.imshow(csf_real[sel_slice, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 4, title = 'Generated Mask'); plt.imshow(csf_gen[sel_slice, :, :], cmap = plt.cm.binary)
            csf_logger.experiment.add_figure(f"Image Results", csf_plot, test_iter); test_iter += 1
            
            # --------------------------------------------------------------------------------------------

            # End of Testing Epoch Image Result Writing
            if len(test_loader) > 1000 and batch_idx > len(test_loader) // 2:
                wm_logger.experiment.add_scalar("MSE_Loss", wm_mse.item(), batch_idx)
                wm_logger.experiment.add_scalar("MAE_Loss", wm_mae.item(), batch_idx)
                wm_logger.experiment.add_scalar("SSIM_Index", np.mean(wm_ssim), batch_idx)
                gm_logger.experiment.add_scalar("MSE_Loss", gm_mse.item(), batch_idx)
                gm_logger.experiment.add_scalar("MAE_Loss", gm_mae.item(), batch_idx)
                gm_logger.experiment.add_scalar("SSIM_Index", np.mean(gm_ssim), batch_idx)
                csf_logger.experiment.add_scalar("MSE_Loss", csf_mse.item(), batch_idx)
                csf_logger.experiment.add_scalar("MAE_Loss", csf_mae.item(), batch_idx)
                csf_logger.experiment.add_scalar("SSIM_Index", np.mean(csf_ssim), batch_idx)
            else:
                wm_logger.experiment.add_scalar("MSE Loss", wm_mse.item(), batch_idx)
                wm_logger.experiment.add_scalar("MAE Loss", wm_mae.item(), batch_idx)
                wm_logger.experiment.add_scalar("SSIM Index", np.mean(wm_ssim), batch_idx)
                gm_logger.experiment.add_scalar("MSE Loss", gm_mse.item(), batch_idx)
                gm_logger.experiment.add_scalar("MAE Loss", gm_mae.item(), batch_idx)
                gm_logger.experiment.add_scalar("SSIM Index", np.mean(gm_ssim), batch_idx)
                csf_logger.experiment.add_scalar("MSE Loss", csf_mse.item(), batch_idx)
                csf_logger.experiment.add_scalar("MAE Loss", csf_mae.item(), batch_idx)
                csf_logger.experiment.add_scalar("SSIM Index", np.mean(csf_ssim), batch_idx)
            #import tensorboard