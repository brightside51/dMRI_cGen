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
from MUDI_fcglVNN import MUDI_fcglVNN

# Full cglVNN Model Class Importing
sys.path.append("Model Builds")
from fcglVNN import fcglVNN
from EarlyStopping import EarlyStopping
#from ReconCallback import ReconCallback

# --------------------------------------------------------------------------------------------

# Result Plotting Functionality (V1)
def plot_results(
    logger: TensorBoardLogger, loss_: str,
    best_idx: int, best_loss: torch.Tensor or np.float,
    worst_idx: int, worst_loss: torch.Tensor or np.float,
    img_best_gt, img_best_fake, img_worst_gt, img_worst_fake,
    patient_id: int = 14, sel_slice: int = 25, epoch = 0
):

    # Set Example Reconstruction Results Image Plotting
    plot = plt.figure(figsize = (20, 22))
    plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
    plt.suptitle(f'Validation Patient {patient_id} | Parameter Results | {loss_} Loss')
    if type(best_loss) == torch.Tensor: best_loss = best_loss.item(); worst_loss= worst_loss.item()

    # Set Example Original & Best Loss Reconstructed Image Subplots
    plt.subplot(2, 2, 1, title = f'Best {loss_} | Original | Parameter #{best_idx}')
    plt.imshow(img_best_gt[0, sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 2, title = f'Best {loss_} | Fake | {loss_}: {np.round(best_loss, 4)}')
    plt.imshow(img_best_fake[0, sel_slice, :, :], cmap = plt.cm.binary)
    
    # Set Example Original & Worst Loss Reconstructed Image Subplots
    plt.subplot(2, 2, 3, title = f'Worst {loss_} | Original | Parameter #{worst_idx}')
    plt.imshow(img_worst_gt[0, sel_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 4, title = f'Worst {loss_} | Fake | {loss_}: {np.round(worst_loss, 4)}')
    plt.imshow(img_worst_fake[0, sel_slice, :, :], cmap = plt.cm.binary)

    # Tensorboard Reconstruction Callback
    logger.experiment.add_figure(f"{loss_} Results", plot, epoch)
    logger.experiment.add_scalar(f"Best {loss_}", best_loss, epoch)
    logger.experiment.add_scalar(f"Worst {loss_}", worst_loss, epoch)
    return logger

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# fcglVNN Model Training Script (V1)
def fcglVNN_train(
    settings,
):

    # Seed Random State for Reproducibility
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    # Experiment Logs Directories Initialization
    checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs')
    train_logger = TensorBoardLogger(checkpoint_folderpath, 'train')
    val_logger = TensorBoardLogger(checkpoint_folderpath, 'validation')

    # Training & Validation DataLoaders Initialization
    train_set = []; val_set = []; viz_logger = []
    train_loader = []; val_loader = []
    for p, patient_id in enumerate(settings.patient_list):

        # Patient Set & DataLoader Initialization
        if patient_id in settings.train_patient_list:
            print(f"Patient #{patient_id} | Training Set:")
            train_set.append(   MUDI_fcglVNN(   settings, subject = [patient_id], mode = 'Train',
                                                target_param = settings.train_target_param,
                                                target_voxel = settings.train_target_voxel))
            train_loader.append(DataLoader(     dataset = train_set[-1], pin_memory = True,
                                                shuffle = settings.train_sample_shuffle,
                                                num_workers = settings.num_workers,
                                                batch_size = settings.batch_size))
        elif patient_id in settings.val_patient_list:
            print(f"Patient #{patient_id} | Validation Set:")
            val_set.append(     MUDI_fcglVNN(   settings, subject = [patient_id], mode = 'Train',
                                                target_param = settings.val_target_param,
                                                target_voxel = settings.val_target_voxel))
            val_loader.append(  DataLoader(     dataset = val_set[-1], pin_memory = True,
                                                shuffle = settings.val_sample_shuffle,
                                                num_workers = settings.num_workers,
                                                batch_size = len(val_set[-1].idxv_target)))
            viz_logger.append(  TensorBoardLogger(checkpoint_folderpath, f'validation/p{patient_id}'))
        else: print(f"Patient #{patient_id} | Test Set:     > Not Included")
    
    # --------------------------------------------------------------------------------------------

    # Model & Optimizer Setup
    print(f"Running\n     > Training fcglVNN Model with {torch.cuda.device_count()} GPUs!")
    model = fcglVNN(settings)#.to(settings.device)
    model = nn.DataParallel(model, device_ids = settings.device_ids).to(settings.device)
    optimizer = torch.optim.AdamW(  model.parameters(), lr = settings.base_lr,
                                    weight_decay = settings.weight_decay)

    # Model Checkpoint Loading
    model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best fcglVNN.pt")
    if model_filepath.exists():
        checkpoint = torch.load(model_filepath, map_location = 'cpu')#)settings.device)
        model.load_state_dict(checkpoint['Model']); optimizer.load_state_dict(checkpoint['Optimizer'])
        save_epoch = checkpoint['Current Epoch']; torch.set_rng_state(checkpoint['RNG State'])
        print(f"     > Loading fcglVNN Model for {settings.model_version}: {save_epoch} Past Epochs")
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
            mask = MUDI_fcglVNN.get_mask(settings, num_patient = patient_id)
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
                train_logger.experiment.add_scalar("MSE Loss", mse_loss.item(), train_iter)
                train_mse_loss.append(mse_loss.item()); train_iter += 1

            # Inter-Patient Reconstruction Parameter Sharing Functionality
            if settings.interpatient_sharing:
                if p == 0: train_set[p].shuffle(); idxh_target_train = train_set[p].idxh_target
                else:
                    train_set[p].shuffle(idxh_target = idxh_target_train)
                    assert(np.all(  train_set[p].idxh_target == train_set[0].idxh_target)
                                    ), f"     > ERROR: Parameter Sharing incorrectly setup!"
            else: train_set[p].shuffle()

        # --------------------------------------------------------------------------------------------

        # Validation Set Reconstruction Loss Checkpoints
        best_mse = 1000; worst_mse = 0
        best_ssim = 0; worst_ssim = 1000

        # Validation Patient Loop
        with torch.no_grad():
            model.eval(); val_ssim_loss = []; val_mse_loss = []
            for p, patient_id in enumerate(settings.val_patient_list):
            
                # Training Iteration Loop
                mask = MUDI_fcglVNN.get_mask(settings, num_patient = patient_id)
                val_bar = tqdm(   enumerate(val_loader[p]), total = len(val_loader[p]),
                    desc = f'Epoch #{epoch} | Validation Patient {patient_id}', unit = 'Batches')
                for batch_idx, batch in val_bar:
            
                    # Batch Handling
                    img_gt = torch.Tensor(unmask(batch[1].reshape((1,
                        len(batch[1]))), mask).get_fdata().T).to(settings.device)

                    # Forward Propagation
                    X_fake = torch.squeeze(model(batch[0].to(settings.device)), dim = 1)
                    img_fake = torch.Tensor(unmask(X_fake.detach().cpu().reshape((1,
                                            len(batch[1]))), mask).get_fdata().T)
                    gc.collect(); torch.cuda.empty_cache()

                    # Loss Computation
                    mse_loss = mse_criterion(X_fake, batch[1].to(settings.device)).detach().cpu().numpy()
                    ssim_loss, ssim_img = ssim( img_gt[0].cpu().numpy().astype(np.float32), 
                                                img_fake[0].cpu().numpy().astype(np.float32), full = True,
                                        data_range = (torch.max(img_gt) - torch.min(img_gt)).cpu().numpy())
                    ssim_loss = np.mean(ssim_loss); del ssim_img, batch, X_fake

                    # Loss Appending 
                    val_logger.experiment.add_scalar("MSE Loss", mse_loss.item(), val_iter)
                    val_logger.experiment.add_scalar("SSIM Index", ssim_loss, val_iter)
                    val_mse_loss.append(mse_loss.item()); val_ssim_loss.append(ssim_loss); val_iter += 1

                    # --------------------------------------------------------------------------------------------

                    # MSE & SSIM Loss Assignation for Parameters
                    param_idx = batch_idx % val_set[p].h_target
                    if mse_loss < best_mse:
                        best_mse = mse_loss
                        best_mse_idx = val_set[p].idxh_target[param_idx]
                        img_fake_best_mse = img_fake.detach().cpu()
                        img_gt_best_mse = img_gt.detach().cpu()
                    if mse_loss > worst_mse:
                        worst_mse = mse_loss
                        worst_mse_idx = val_set[p].idxh_target[param_idx]
                        img_fake_worst_mse = img_fake.detach().cpu()
                        img_gt_worst_mse = img_gt.detach().cpu()
                    if ssim_loss > best_ssim:
                        best_ssim = ssim_loss
                        best_ssim_idx = val_set[p].idxh_target[param_idx]
                        img_fake_best_ssim = img_fake.detach().cpu()
                        img_gt_best_ssim = img_gt.detach().cpu()
                    if ssim_loss < worst_ssim:
                        worst_ssim = ssim_loss
                        worst_ssim_idx = val_set[p].idxh_target[param_idx]
                        img_fake_worst_ssim = img_fake.detach().cpu()
                        img_gt_worst_ssim = img_gt.detach().cpu()
                    gc.collect(); torch.cuda.empty_cache(); del img_gt, img_fake

                # Inter-Patient Reconstruction Parameter Sharing Functionality
                if settings.interpatient_sharing:
                    if p == 0: val_set[p].shuffle(); idxh_target_val = val_set[p].idxh_target
                    else:
                        val_set[p].shuffle(idxh_target = idxh_target_val)
                        assert(np.all(  val_set[p].idxh_target == val_set[0].idxh_target)
                                        ), f"     > ERROR: Parameter Sharing incorrectly setup!"
                else: val_set[p].shuffle()

        # --------------------------------------------------------------------------------------------

                # MSE Epoch Results
                viz_logger[p] = plot_results(   epoch = epoch, loss_ = 'MSE',
                                                logger = viz_logger[p], sel_slice = settings.sel_slice,
                                                best_idx = best_mse_idx, worst_idx = worst_mse_idx,
                                                best_loss = best_mse, worst_loss = worst_mse,
                                                img_best_gt = img_gt_best_mse, img_best_fake = img_fake_best_mse,
                                                img_worst_gt = img_gt_worst_mse, img_worst_fake = img_fake_worst_mse)
                del best_mse_idx, img_gt_best_mse, img_fake_best_mse, best_mse,\
                    worst_mse_idx, img_gt_worst_mse, img_fake_worst_mse, worst_mse

                # SSIM Epoch Results
                viz_logger[p] = plot_results(   epoch = epoch, loss_ = 'SSIM',
                                                logger = viz_logger[p], sel_slice = settings.sel_slice,
                                                best_idx = best_ssim_idx, worst_idx = worst_ssim_idx,
                                                best_loss = best_ssim, worst_loss = worst_ssim,
                                                img_best_gt = img_gt_best_ssim, img_best_fake = img_fake_best_ssim,
                                                img_worst_gt = img_gt_worst_ssim, img_worst_fake = img_fake_worst_ssim)
                del best_ssim_idx, img_gt_best_ssim, img_fake_best_ssim, best_ssim,\
                    worst_ssim_idx, img_gt_worst_ssim, img_fake_worst_ssim, worst_ssim
                
        # End of Epoch Mean Loss Writing
        train_mse_loss = np.mean(np.array(train_mse_loss))
        val_mse_loss = np.mean(np.array(val_mse_loss))
        val_ssim_loss = np.mean(np.array(val_ssim_loss))
        train_logger.experiment.add_scalar("Mean MSE Loss", train_mse_loss, epoch)
        val_logger.experiment.add_scalar("Mean MSE Loss", val_mse_loss, epoch)        
        val_logger.experiment.add_scalar("Mean SSIM Index", val_ssim_loss, epoch)
        
        # --------------------------------------------------------------------------------------------

        # Early Stopping Callback Application
        early_stop = earlyStopping( loss = val_mse_loss, epoch = epoch,
                                    model = model, optimizer = optimizer)
        if early_stop: print(f'     > Training Finished at Epoch #{epoch}'); return
