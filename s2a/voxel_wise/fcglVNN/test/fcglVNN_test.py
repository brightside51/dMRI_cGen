# Library Imports
import os
import gc
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import nibabel as nib
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
from MUDI_fcglVNN import MUDI_fcglVNN

# Full cglVNN Model Class Importing
sys.path.append("Model Builds")
from fcglVNN import fcglVNN
from EarlyStopping import EarlyStopping
#from ReconCallback import ReconCallback

# --------------------------------------------------------------------------------------------

# fcglVNN Model Testing Script (V1)
def fcglVNN_test(
    settings,
    patient_id: int = 15,
    mode: str = 'Train'
):
        
    # Seed Random State for Reproducibility
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    # Model & Optimizer Setup
    print(f"Evaluation\n     > Testing fcglVNN Model with {torch.cuda.device_count()} GPUs!")
    model = fcglVNN(settings)#.to(settings.device)
    model = nn.DataParallel(model, device_ids = settings.device_ids).to(settings.device)
    #optimizer = torch.optim.AdamW(  model.parameters(), lr = settings.base_lr,
    #                                weight_decay = settings.weight_decay)

    # Model Checkpoint Loading
    model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best fcglVNN.pt")
    assert(model_filepath.exists()), f"ERROR: fcglVNN Model (V{settings.model_version}) not Found!"
    checkpoint = torch.load(model_filepath, map_location = settings.device)
    model.load_state_dict(checkpoint['Model'])
    #optimizer.load_state_dict(checkpoint['Optimizer'])
    save_epoch = checkpoint['Current Epoch']
    torch.set_rng_state(checkpoint['RNG State'])

    # Experiment Logs Directories Initialization
    checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs')
    mse_criterion = nn.MSELoss(reduction = 'mean'); del checkpoint
    test_logger = TensorBoardLogger(checkpoint_folderpath, f'test/epoch{save_epoch}/p{patient_id}/{mode}')
    print(f"     > Evaluating fcglVNN Model for Version #{settings.model_version}: {save_epoch} Past Epochs")

    # --------------------------------------------------------------------------------------------

    # DataSet & DataLoader Initialization
    mask = MUDI_fcglVNN.get_mask(settings, num_patient = patient_id)
    test_set = MUDI_fcglVNN(    settings, subject = [patient_id], mode = mode,
                                target_param = settings.val_target_param,
                                target_voxel = settings.val_target_voxel)
    test_loader = DataLoader(   dataset = test_set, shuffle = False,
                                num_workers = 0,#settings.num_workers,
                                batch_size = len(test_set.idxv_target),
                                pin_memory = False)
    
    img_fake = np.empty((len(test_set.idxh_target), mask.shape[2], mask.shape[1], mask.shape[0]))
    img_mse = np.empty((len(test_set.idxh_target), mask.shape[2], mask.shape[1], mask.shape[0]))
    img_ssim = np.empty((len(test_set.idxh_target), mask.shape[2], mask.shape[1], mask.shape[0]))

    # Batch Iteration Loop
    with torch.no_grad():
        model.eval()
        test_bar = tqdm(   enumerate(test_loader), total = len(test_loader),
            desc = f'Test Patient {patient_id}', unit = 'Batches')
        for batch_idx, batch in test_bar:

            # Forward Propagation
            img_gt = torch.Tensor(unmask(batch[1].reshape((1,
                        len(batch[1]))), mask).get_fdata().T).to(settings.device)
            X_fake = torch.squeeze(model(batch[0].to(settings.device)), dim = 1)
            img_fake[batch_idx] = torch.Tensor(unmask(X_fake.detach().cpu().reshape((1,
                                    len(batch[1]))), mask).get_fdata().T).cpu().numpy()
            gc.collect(); torch.cuda.empty_cache()

            # Loss Computation
            mse_loss = mse_criterion(X_fake, batch[1].to(settings.device)).detach().cpu().numpy()
            ssim_loss, img_ssim[batch_idx] = ssim(  img_gt[0].cpu().numpy().astype(np.float32), 
                                                    img_fake[batch_idx].astype(np.float32), full = True,
                                            data_range = (torch.max(img_gt) - torch.min(img_gt)).cpu().numpy())
            img_mse[batch_idx] = (img_fake[batch_idx] - img_gt[0].cpu().numpy()) ** 2
            ssim_loss = np.mean(ssim_loss); del batch, X_fake
            
            # --------------------------------------------------------------------------------------------

            # Target Parameter Plot Initialization
            param_target = test_set.idxh_target[batch_idx % test_set.h_target]; test_plot = plt.figure(figsize = (20, 22))
            plt.suptitle(f'Test Patient #{patient_id} | Parameter #{param_target} | Slice #{settings.sel_slice}' +
                                                f'\nMSE: {np.round(mse_loss, 5)} | SSIM: {np.round(ssim_loss, 5)}\n')
            plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout(); gc.collect(); torch.cuda.empty_cache()

            # Original, Reconstruction & Loss Heatmap Plotting
            plt.subplot(2, 2, 1, title = 'Original Scan'); plt.imshow(img_gt[0, settings.sel_slice, :, :].cpu(), cmap = plt.cm.binary)
            plt.subplot(2, 2, 2, title = 'Reconstructed Scan'); plt.imshow(img_fake[batch_idx, settings.sel_slice, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 3, title = 'MSE Loss Heatmap'); plt.imshow(img_mse[batch_idx, settings.sel_slice, :, :], cmap = 'hot')
            plt.subplot(2, 2, 4, title = 'SSIM Index Mask'); plt.imshow(img_ssim[batch_idx, settings.sel_slice, :, :], cmap = plt.cm.binary)
            
            # Tensorboard Reconstruction Callback
            test_logger.experiment.add_figure(f"Target Results", test_plot, batch_idx)
            if mode == 'Train' and len(test_loader) > 1000:
                if batch_idx < len(test_loader) // 2:
                    test_logger.experiment.add_scalar(f"MSE Loss", mse_loss, batch_idx)
                    test_logger.experiment.add_scalar(f"SSIM Index", ssim_loss, batch_idx)
                else:
                    test_logger.experiment.add_scalar(f"MSE_Loss", mse_loss, batch_idx)
                    test_logger.experiment.add_scalar(f"SSIM_Index", ssim_loss, batch_idx)
            else:
                test_logger.experiment.add_scalar(f"MSE Loss", mse_loss, batch_idx)
                test_logger.experiment.add_scalar(f"SSIM Index", ssim_loss, batch_idx)
    
    # Result Image Saving
    img_fake = nib.Nifti1Image(img_fake.T, affine = np.eye(4)); img_fake.header.get_xyzt_units()
    img_mse = nib.Nifti1Image(img_mse.T, affine = np.eye(4)); img_mse.header.get_xyzt_units()
    img_ssim = nib.Nifti1Image(img_ssim.T, affine = np.eye(4)); img_ssim.header.get_xyzt_units()
    nib.save(img_fake, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/{mode}/img_fake.nii.gz"))
    nib.save(img_mse, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/{mode}/img_mse.nii.gz"))
    nib.save(img_ssim, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/{mode}/img_ssim.nii.gz"))
