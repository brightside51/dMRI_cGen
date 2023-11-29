# Library Imports
import os
import gc
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pathlib import Path
from nilearn.masking import unmask
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from pytorch_lightning.loggers import TensorBoardLogger
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_fcNN import MUDI_fcNN

# Full cglVNN Model Class Importing
sys.path.append("Model Builds")
from fcNN import fcNN
from EarlyStopping import EarlyStopping
#from ReconCallback import ReconCallback

# --------------------------------------------------------------------------------------------

# Result Plotting Functionality (V0)
def plot_results(
    logger: TensorBoardLogger,
    best_info: dict,
    worst_info: dict,
    patient_id: int = 14,
    sel_slice: int = 25,
    epoch = 0,
    mode: str = 'Train',
    loss: str = 'MSE Loss'
):

    # Set Example Reconstruction Results Image Plotting
    plot = plt.figure(figsize = (20, 25)); plt.suptitle(f"Overall {mode} | {loss}")
    plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()

    # Set Example Original & Best Loss Reconstructed Image Subplots
    plt.subplot(2, 2, 1, title =        f"Best | Target Parameter #{best_info['idxh']}")
    plt.imshow(best_info['img_gt'][     sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 2, title =        f"Best | Reconstruction | {loss}: {np.round(best_info['loss'], 5)}")
    plt.imshow(best_info['img_fake'][   sel_slice, :, :], cmap = plt.cm.binary)

    # Set Example Original & Worst Loss Reconstructed Image Subplots
    plt.subplot(2, 2, 3, title =        f"Best | Target Parameter #{worst_info['idxh']}")
    plt.imshow(worst_info['img_gt'][    sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 4, title =        f"Best | Reconstruction | {loss}: {np.round(worst_info['loss'], 5)}")
    plt.imshow(worst_info['img_fake'][  sel_slice, :, :], cmap = plt.cm.binary)

    # Tensorboard Reconstruction Callback
    logger.experiment.add_figure(f"{mode} Image Results", plot, epoch)
    logger.experiment.add_scalar("Best Loss", best_info['loss'], epoch)
    logger.experiment.add_scalar("Worst Loss", worst_info['loss'], epoch)
    return logger


# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# fcNN Model Training Script (V0)
def fcNN_train(
    settings,
):

    # Seed Random State for Reproducibility
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    # Experiment Logs Directories Initialization
    checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs')
    train_logger = TensorBoardLogger(checkpoint_folderpath, 'train')
    val_mse_logger = TensorBoardLogger(checkpoint_folderpath, 'validation/mse')
    val_ssim_logger = TensorBoardLogger(checkpoint_folderpath, 'validation/ssim')

    # Training & Validation DataLoaders Initialization
    train_set = []; val_set = []
    train_loader = []; val_loader = []
    for p, patient_id in enumerate(settings.patient_list):

        # Patient Set & DataLoader Initialization
        if patient_id in settings.train_patient_list:
            print(f"Patient #{patient_id} | Training Set:")
            train_set.append(   MUDI_fcNN(  settings, subject = [patient_id],
                                            target_voxel = settings.train_target_voxel))
            train_loader.append(DataLoader( dataset = train_set[-1], pin_memory = True,
                                            shuffle = settings.train_sample_shuffle,
                                            num_workers = settings.num_workers,
                                            batch_size = settings.batch_size))
        elif patient_id in settings.val_patient_list:
            print(f"Patient #{patient_id} | Validation Set:")
            val_set.append(     MUDI_fcNN(  settings, subject = [patient_id],
                                            target_voxel = settings.val_target_voxel))
        else: print(f"Patient #{patient_id} | Test Set:\n     > Not Included")
    
    # --------------------------------------------------------------------------------------------

    # Model & Optimizer Setup
    print(f"Running\n     > Training fcNN Model with {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(fcNN(settings), device_ids = settings.device_ids).to(settings.device)
    optimizer = torch.optim.AdamW(  model.parameters(), lr = settings.base_lr,
                                    weight_decay = settings.weight_decay)

    # Model Checkpoint Loading
    model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best fcNN.pt")
    if model_filepath.exists():
        checkpoint = torch.load(model_filepath, map_location = 'cpu')#)settings.device)
        model.load_state_dict(checkpoint['Model']); optimizer.load_state_dict(checkpoint['Optimizer'])
        save_epoch = checkpoint['Current Epoch']; torch.set_rng_state(checkpoint['RNG State'])
        print(f"     > Loading fcNN Model for {settings.model_version}: {save_epoch} Past Epochs")
        del checkpoint
    else: save_epoch = -1

    # Criterion & Early Stopping Setup
    mse_criterion = nn.MSELoss(reduction = 'mean')
    earlyStopping = EarlyStopping(settings)
    
    # --------------------------------------------------------------------------------------------
    
    # Epoch Iteration Loop
    train_iter = 0; val_iter = 0
    for epoch in range(save_epoch + 1, settings.num_epochs):

        # Training Patient Loop
        train_mse_loss = []; print(f"Training Epoch #{epoch}:")
        for p, patient_id in enumerate(settings.train_patient_list):

            # Training Iteration Loop
            #mask = MUDI_fcNN.get_mask(settings, num_patient = patient_id)
            train_bar = tqdm(   enumerate(train_loader[p]), total = len(train_loader[p]),
                desc = f'Epoch #{epoch} | Training Patient {patient_id}', unit = 'Batches')
            for batch_idx, batch in train_bar:

                # Forward Propagation
                model.train(); model.zero_grad(); optimizer.zero_grad()
                X_fake = model(batch[0].to(settings.device))
                X_fake = torch.squeeze(X_fake, dim = 1)
                gc.collect(); torch.cuda.empty_cache()

                # Backward Propagation
                mse_loss = mse_criterion(X_fake, batch[1].to(settings.device))
                mse_loss.backward(); optimizer.step(); del batch, X_fake
                train_logger.experiment.add_scalar("Batch Loss", mse_loss.item(), train_iter)
                train_mse_loss.append(mse_loss.item()); train_iter += 1

            # Inter-Patient Reconstruction Parameter Sharing Functionality
            if settings.interpatient_sharing:
                if p == 0: train_set[p].shuffle(); idxv_target_train = train_set[p].idxv_target
                else:
                    train_set[p].shuffle(idxv_target = idxv_target_train)
                    assert(np.all(  train_set[p].idxv_target == train_set[0].idxv_target)
                                    ), f"     > ERROR: Parameter Sharing incorrectly setup!"
            else: train_set[p].shuffle()

        # --------------------------------------------------------------------------------------------

        # Validation Set Reconstruction Loss Checkpoints
        best_mse = 1000; worst_mse = 0
        best_ssim = 0; worst_ssim = 1000
        print(f"Validation Epoch #{epoch}:")

        # Validation Patient Loop
        with torch.no_grad():
            model.eval(); val_ssim_loss = []; val_mse_loss = []
            for p, patient_id in enumerate(settings.val_patient_list):
                        
                # Batch Handling
                mask = MUDI_fcNN.get_mask(settings, num_patient = patient_id)
                X_train = torch.Tensor(val_set[p].data[val_set[p].idxv_target, :].T)
                img_gt = torch.Tensor(unmask(X_train, mask).get_fdata().T).to(settings.device)

                # Forward Propagation
                X_fake = model(X_train[val_set[p].idxh_train].T.to(settings.device))
                X_fake = torch.squeeze(X_fake, dim = 1).T
                img_fake = torch.Tensor(unmask(X_fake.detach().cpu().reshape((settings.out_channels,
                                                        val_set[p].v_target)), mask).get_fdata().T)
                gc.collect(); torch.cuda.empty_cache()

                # --------------------------------------------------------------------------------------------

                # Parameter-Wise MSE & SSIM Loss Assignation
                val_bar = tqdm(   enumerate(np.array(val_set[p].params)), total = settings.out_channels,
                            desc = f'Epoch #{epoch} | Validation Patient {patient_id}', unit = 'Batches')
                for i, idxh_target in val_bar:

                    # Parameter-Wise Loss Computation
                    mse_loss = mse_criterion(   X_fake[i, :].detach().cpu(),
                                                X_train[i, :].detach().cpu())
                    ssim_loss, ssim_img = ssim( img_gt[i].detach().cpu().numpy().astype(np.float32), 
                                                img_fake[i].detach().cpu().numpy().astype(np.float32), full = True,
                                        data_range = (torch.max(img_gt[i]) - torch.min(img_gt[i])).cpu().numpy())
                    
                    # Parameter-Wise Loss Appending
                    val_mse_logger.experiment.add_scalar("Batch Loss", mse_loss.item(), val_iter)
                    val_ssim_logger.experiment.add_scalar("Batch Loss", ssim_loss, val_iter)
                    val_mse_loss.append(mse_loss.item()); val_ssim_loss.append(ssim_loss); val_iter += 1

                    # Best & Worst MSE Result Saving
                    if mse_loss.item() < best_mse:
                        best_mse = mse_loss.item()
                        best_mse_info = {   'loss': mse_loss.item(), 'idxh': i,
                                            'img_gt': img_gt[i].detach().cpu(),
                                            'img_fake': img_fake[i].detach().cpu()}
                    if mse_loss.item() > worst_mse:
                        worst_mse = mse_loss.item()
                        worst_mse_info = {  'loss': mse_loss.item(), 'idxh': i,
                                            'img_gt': img_gt[i].detach().cpu(),
                                            'img_fake': img_fake[i].detach().cpu()}

                    # Best & Worst SSIM Result Saving
                    if ssim_loss > best_ssim:
                        best_ssim = ssim_loss
                        best_ssim_info = {  'loss': ssim_loss, 'idxh': i,
                                            'img_gt': img_gt[i].detach().cpu(),
                                            'img_fake': img_fake[i].detach().cpu()}
                    if ssim_loss < worst_ssim:
                        worst_ssim = ssim_loss
                        worst_ssim_info = { 'loss': ssim_loss, 'idxh': i,
                                            'img_gt': img_gt[i].detach().cpu(),
                                            'img_fake': img_fake[i].detach().cpu()}
                gc.collect(); torch.cuda.empty_cache(); del X_fake, X_train, img_gt, img_fake, ssim_img

                # Inter-Patient Reconstruction Parameter Sharing Functionality
                if settings.interpatient_sharing:
                    if p == 0: val_set[p].shuffle(); idxv_target_val = val_set[p].idxv_target
                    else:
                        val_set[p].shuffle(idxv_target = idxv_target_val)
                        assert(np.all(  val_set[p].idxv_target == val_set[0].idxv_target)
                                        ), f"     > ERROR: Parameter Sharing incorrectly setup!"
                else: val_set[p].shuffle()

        # --------------------------------------------------------------------------------------------

                # MSE Epoch Results
                val_mse_logger = plot_results(  logger = val_mse_logger,
                                                best_info = best_mse_info,
                                                worst_info = worst_mse_info,
                                                sel_slice = settings.sel_slice,
                                                mode = 'Validation', loss = 'MSE Loss',
                                                patient_id = patient_id, epoch = epoch)

                # SSIM Epoch Results
                val_ssim_logger = plot_results( logger = val_ssim_logger,
                                                best_info = best_ssim_info,
                                                worst_info = worst_ssim_info,
                                                sel_slice = settings.sel_slice,
                                                mode = 'Validation', loss = 'SSIM Index',
                                                patient_id = patient_id, epoch = epoch)
                del best_mse_info, worst_mse_info, best_ssim_info, worst_ssim_info
                
        # End of Epoch Mean Loss Writing
        train_mse_loss = np.mean(np.array(train_mse_loss))
        val_mse_loss = np.mean(np.array(val_mse_loss))
        val_ssim_loss = np.mean(np.array(val_ssim_loss))
        train_logger.experiment.add_scalar("Mean Loss", train_mse_loss, epoch)
        val_mse_logger.experiment.add_scalar("Mean Loss", val_mse_loss, epoch)        
        val_ssim_logger.experiment.add_scalar("Mean Loss", val_ssim_loss, epoch)
        
        # --------------------------------------------------------------------------------------------

        # Early Stopping Callback Application
        early_stop = earlyStopping( loss = val_mse_loss, epoch = epoch,
                                    model = model, optimizer = optimizer)
        train_logger.experiment.add_scalar("Early Stopping Counter", earlyStopping.counter, epoch)
        if early_stop: print(f'     > Training Finished at Epoch #{epoch}'); return
