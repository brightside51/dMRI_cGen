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
import time
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
from MUDI_cVNP import MUDI_cVNP

# Full cglVNN Model Class Importing
sys.path.append("Model Builds")
from cVNP import cVNP
#from ReconCallback import ReconCallback

# --------------------------------------------------------------------------------------------

# Result Plotting Functionality (V0)
def ResultCallback(
    logger: TensorBoardLogger,
    best_info: dict,
    worst_info: dict,
    epoch: int = 0,
    num_slice: int = 25,
    mode: str = 'Test',
    loss: str = 'MSE Loss',
):

    # Set Example Reconstruction Results Image Plotting
    plot = plt.figure(figsize = (30, 25)); plt.suptitle(f"Overall {mode} | {loss}")
    plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()

    # Set Example Original & Best Loss Reconstructed Image Subplots
    plt.subplot(2, 3, 1, title =        f"Best | Source Parameter #{best_info['idxh_source']}")
    plt.imshow(best_info['img_source'][ num_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 2, title =        f"Best | Target Parameter #{best_info['idxh_target']}")
    plt.imshow(best_info['img_target'][ num_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 3, title =        f"Best | Reconstruction | {loss}: {np.round(best_info['loss'], 5)}")
    plt.imshow(best_info['img_fake'][   0, num_slice, :, :], cmap = plt.cm.binary)

    # Set Example Original & Worst Loss Reconstructed Image Subplots
    plt.subplot(2, 3, 4, title =        f"Worst | Source Parameter #{worst_info['idxh_source']}")
    plt.imshow(worst_info['img_source'][num_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 5, title =        f"Worst | Target Parameter #{worst_info['idxh_target']}")
    plt.imshow(worst_info['img_target'][num_slice, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 6, title =        f"Worst | Reconstruction | {loss}: {np.round(worst_info['loss'], 5)}")
    plt.imshow(worst_info['img_fake'][  0, num_slice, :, :], cmap = plt.cm.binary)

    # Tensorboard Reconstruction Callback
    logger.experiment.add_figure(f"{mode} Image Results", plot, epoch)
    logger.experiment.add_scalar("Best Loss", best_info['loss'], epoch)
    logger.experiment.add_scalar("Worst Loss", worst_info['loss'], epoch)
    return logger

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# cVNP Model Testing Script (V0)
def cVNP_test(
    settings,
    patient_id: int = 15
):
        
    # Seed Random State for Reproducibility
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    # cVNP Model Loading
    print(f"Evaluation\n     > Testing cVNP Model with {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(cVNP(settings)).to(settings.device)
    model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best cVNP.pt")
    assert(model_filepath.exists()), f"ERROR: cVNP Model (V{settings.model_version}) not Found!"
    checkpoint = torch.load(model_filepath, map_location = settings.device)
    model.load_state_dict(checkpoint['Model'])
    #optimizer.load_state_dict(checkpoint['Optimizer'])
    save_epoch = checkpoint['Current Epoch']
    #torch.set_rng_state(checkpoint['RNG State'])
    mse_criterion = nn.MSELoss(reduction = 'mean'); del checkpoint

    # Experiment Logs Directories Initialization
    checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs')
    test_logger = TensorBoardLogger(checkpoint_folderpath, f'test/epoch{save_epoch}/p{patient_id}')
    mse_logger = TensorBoardLogger(checkpoint_folderpath, f'test/epoch{save_epoch}/p{patient_id}/mse')
    ssim_logger = TensorBoardLogger(checkpoint_folderpath, f'test/epoch{save_epoch}/p{patient_id}/ssim')
    print(f"     > Evaluating fcglVNN Model for Version #{settings.model_version}: {save_epoch} Past Epochs")

    # --------------------------------------------------------------------------------------------

    # Test Set Initialization
    mask = MUDI_cVNP.get_mask(settings, num_patient = 14)
    testset = MUDI_cVNP(    settings, subject = [14], random = False,
                            source_param = 100, target_param = 100,
                            target_voxel = 100, param_loop = 100)
    testloader = DataLoader(dataset = testset, shuffle = False,
                            num_workers = 0, pin_memory = False,
                            batch_size = len(testset.idxv_target))

    # Loss Value Initialization
    test_mse = []; test_ssim = []; test_iter = 0
    best_mse = 1000; worst_mse = 0
    best_ssim = 0; worst_ssim = 1000
    
    # Batch Iteration Loop
    with torch.no_grad():
        test_bar = tqdm(   enumerate(testloader), total = len(testloader),
            desc = f'Test Patient {patient_id}', unit = 'Batches'); model.eval()
        for batch_idx, batch in test_bar:

            # Forward Propagation
            idxh_source = batch['param_source']; idxh_target = batch['param_target']
            X_fake = model( batch['X_train'].to(settings.device),
                            batch['y_train'].to(settings.device),
                            batch['y_target'].to(settings.device))

            # Image Handling
            img_fake = torch.Tensor(unmask(X_fake.detach().cpu().T, mask).get_fdata().T)
            img_source = torch.Tensor(unmask(batch['X_train'], mask).get_fdata().T)
            img_target = torch.Tensor(unmask(batch['X_target'], mask).get_fdata().T)

            # Loss Computation
            mse_loss = mse_criterion(X_fake.T[0], batch['X_target'].to(settings.device))
            ssim_loss, ssim_img = ssim( img_target.detach().cpu().numpy().astype(np.float32), 
                                        img_fake[0].cpu().numpy().astype(np.float32), full = True,
                                        data_range = (torch.max(img_target) - torch.min(img_target)).cpu().numpy())
            ssim_loss = np.mean(ssim_loss); del ssim_img, X_fake, batch

            # Loss Appending
            test_mse.append(mse_loss.item()); test_ssim.append(ssim_loss)
            mse_logger.experiment.add_scalar("Batch Loss", mse_loss.item(), test_iter)
            ssim_logger.experiment.add_scalar("Batch Loss", ssim_loss, test_iter)

            # --------------------------------------------------------------------------------------------

            # Set Example Reconstruction Results Image Plotting
            plot = plt.figure(figsize = (15, 25)); plt.suptitle(f"Source #{idxh_source} -> Target #{idxh_target}")
            plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
            plt.subplot(1, 3, 1, title = f"Best | Source Parameter #{idxh_source}")
            plt.imshow(img_source[  settings.sel_slice, :, :], cmap = plt.cm.binary)
            plt.subplot(1, 3, 2, title = f"Best | Target Parameter #{idxh_target}")
            plt.imshow(img_target[  settings.sel_slice, :, :], cmap = plt.cm.binary)
            plt.subplot(1, 3, 3, title = f"Best | Reconstruction | MSE: {np.round(mse_loss.item(), 5)}; SSIM: {np.round(ssim_loss, 5)}")
            plt.imshow(img_fake[    0, settings.sel_slice, :, :], cmap = plt.cm.binary)
            test_logger.experiment.add_figure(f"Batch Image Results", plot, test_iter); test_iter += 1
            
            # --------------------------------------------------------------------------------------------

            # MSE & SSIM Loss Assignation for Parameters
            if mse_loss.item() < best_mse:
                best_mse = mse_loss.item()
                best_mse_info = {   'loss': mse_loss.item(), 'img_fake': img_fake,
                                    'idxh_source': idxh_source, 'idxh_target': idxh_target,
                                    'img_source': img_source, 'img_target': img_target}
            if mse_loss.item() > worst_mse:
                worst_mse = mse_loss.item()
                worst_mse_info = {  'loss': mse_loss.item(), 'img_fake': img_fake,
                                    'idxh_source': idxh_source, 'idxh_target': idxh_target,
                                    'img_source': img_source, 'img_target': img_target}
            if ssim_loss > best_ssim:
                best_ssim = ssim_loss
                best_ssim_info = {  'loss': ssim_loss, 'img_fake': img_fake,
                                    'idxh_source': idxh_source, 'idxh_target': idxh_target,
                                    'img_source': img_source, 'img_target': img_target}
            if ssim_loss < worst_ssim:
                worst_ssim = ssim_loss
                worst_ssim_info = { 'loss': ssim_loss, 'img_fake': img_fake,
                                    'idxh_source': idxh_source, 'idxh_target': idxh_target,
                                    'img_source': img_source, 'img_target': img_target}
            gc.collect(); torch.cuda.empty_cache()
            del img_source, img_target, img_fake, idxh_source, idxh_target

            # --------------------------------------------------------------------------------------------

            # Source Parameter Based Result Appending
            if (batch_idx + 1) % testset.h_target == 0:

                # End of Source Parameter Image Result Writing
                mse_logger = ResultCallback(    logger = mse_logger, mode = 'Test', loss = 'MSE Loss',
                                                epoch = batch_idx // testset.h_target,
                                                best_info = best_mse_info, worst_info = worst_mse_info)
                ssim_logger = ResultCallback(   logger = ssim_logger, mode = 'Test', loss = 'SSIM Index',
                                                epoch = batch_idx // testset.h_target,
                                                best_info = best_ssim_info, worst_info = worst_ssim_info)
                del best_mse_info, worst_mse_info, best_ssim_info, worst_ssim_info

                # Loss Value Re-Initialization
                mse_logger.experiment.add_scalar("Mean Loss", np.mean(test_mse), batch_idx // testset.h_target)
                ssim_logger.experiment.add_scalar("Mean Loss", np.mean(test_ssim), batch_idx // testset.h_target)
                test_mse = []; test_ssim = []; best_mse = 1000; worst_mse = 0; best_ssim = 0; worst_ssim = 1000
    
    """
    # Result Image Saving
    img_fake = nib.Nifti1Image(img_fake.T, affine = np.eye(4)); img_fake.header.get_xyzt_units()
    img_mse = nib.Nifti1Image(img_mse.T, affine = np.eye(4)); img_mse.header.get_xyzt_units()
    img_ssim = nib.Nifti1Image(img_ssim.T, affine = np.eye(4)); img_ssim.header.get_xyzt_units()
    nib.save(img_fake, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/img_fake.nii.gz"))
    nib.save(img_mse, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/img_mse.nii.gz"))
    nib.save(img_ssim, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/img_ssim.nii.gz"))
    """
