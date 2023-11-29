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
from MUDI_fcNN import MUDI_fcNN

# Full cglVNN Model Class Importing
sys.path.append("Model Builds")
from fcNN import fcNN
#from ReconCallback import ReconCallback

# --------------------------------------------------------------------------------------------

# fcglVNN Model Testing Script (V1)
def fcNN_test(
    settings,
    patient_id: int = 15
):
        
    # Seed Random State for Reproducibility
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    random.seed(settings.seed)

    # Model & Optimizer Setup
    print(f"Evaluation\n     > Testing fcNN Model with {torch.cuda.device_count()} GPUs!")
    model = fcNN(settings)#.to(settings.device)
    model = nn.DataParallel(model, device_ids = settings.device_ids).to(settings.device)
    #optimizer = torch.optim.AdamW(  model.parameters(), lr = settings.base_lr,
    #                                weight_decay = settings.weight_decay)

    # Model Checkpoint Loading
    model_filepath = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best fcNN.pt")
    assert(model_filepath.exists()), f"ERROR: fcNN Model (V{settings.model_version}) not Found!"
    checkpoint = torch.load(model_filepath, map_location = settings.device)
    model.load_state_dict(checkpoint['Model'])
    #optimizer.load_state_dict(checkpoint['Optimizer'])
    save_epoch = checkpoint['Current Epoch']
    torch.set_rng_state(checkpoint['RNG State'])

    # Experiment Logs Directories Initialization
    checkpoint_folderpath = os.path.join(f'{settings.modelsave_folderpath}/V{settings.model_version}', 'logs')
    mse_criterion = nn.MSELoss(reduction = 'mean'); del checkpoint
    test_logger = TensorBoardLogger(checkpoint_folderpath, f'test/epoch{save_epoch}/p{patient_id}')
    print(f"     > Evaluating fcNN Model for Version #{settings.model_version}: {save_epoch} Past Epochs")

    # --------------------------------------------------------------------------------------------

    # DataSet & DataLoader Initialization
    mask = MUDI_fcNN.get_mask(  settings, num_patient = patient_id)
    test_set = MUDI_fcNN(       settings, subject = [patient_id],
                                target_voxel = 100)#settings.train_target_voxel)
    test_loader = DataLoader(   dataset = test_set, pin_memory = False,
                                shuffle = False,#settings.train_sample_shuffle,
                                num_workers = 0,#settings.num_workers,
                                batch_size = len(test_set.idxv_target))
    img_fake = np.empty((settings.out_channels, mask.shape[2], mask.shape[1], mask.shape[0]))
    img_mse = np.empty((settings.out_channels, mask.shape[2], mask.shape[1], mask.shape[0]))
    img_ssim = np.empty((settings.out_channels, mask.shape[2], mask.shape[1], mask.shape[0]))    

    # Forward Propagation
    with torch.no_grad():
        model.eval(); batch = next(iter(test_loader))
        #img_gt = torch.Tensor(unmask(batch[1].reshape((settings.out_channels,
            #len(test_set.idxv_target))), mask).get_fdata().T).to(settings.device)
        X_fake = torch.squeeze(model(batch[0].to(settings.device)), dim = 1)
        #img_fake = torch.Tensor(unmask(X_fake.detach().cpu().reshape((settings.out_channels,
                            #len(test_set.idxv_target))), mask).get_fdata().T).cpu().numpy()
        gc.collect(); torch.cuda.empty_cache()

        # Batch Iteration Loop
        test_bar = tqdm(enumerate(np.arange(settings.out_channels)),
                        total = settings.out_channels, unit = 'Batches',
                        desc = f'Test Patient {patient_id}')
        for batch_idx, i in test_bar:

            # Loss Computation
            img_gt = torch.Tensor(unmask(batch[1][:, i].reshape((1,
                            len(test_set.idxv_target))), mask).get_fdata().T).to(settings.device)
            img_fake[i] = torch.Tensor(unmask(X_fake[:, i].detach().cpu().reshape((1,
                            len(test_set.idxv_target))), mask).get_fdata().T).cpu().numpy()
            mse_loss = mse_criterion(X_fake[:, i], batch[1][:, i].to(settings.device)).detach().cpu().numpy()
            ssim_loss, img_ssim[i] = ssim(  img_gt[0].cpu().numpy().astype(np.float32), 
                                            img_fake[i].astype(np.float32), full = True,
                    data_range = (torch.max(img_gt) - torch.min(img_gt)).cpu().numpy())
            img_mse[i] = (img_fake[i] - img_gt.cpu().numpy()) ** 2

            # --------------------------------------------------------------------------------------------

            # Target Parameter Plot Initialization
            test_plot = plt.figure(figsize = (20, 22))
            plt.suptitle(f'Test Patient #{patient_id} | Parameter #{i} | Slice #{settings.sel_slice}' +
                            f'\nMSE: {np.round(mse_loss.item(), 5)} | SSIM: {np.round(ssim_loss, 5)}\n')
            plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
            gc.collect(); torch.cuda.empty_cache()

            # Original, Reconstruction & Loss Heatmap Plotting
            plt.subplot(2, 2, 1, title = 'Original Scan'); plt.imshow(img_gt[0, settings.sel_slice, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 2, title = 'Reconstructed Scan'); plt.imshow(img_fake[i, settings.sel_slice, :, :], cmap = plt.cm.binary)
            plt.subplot(2, 2, 3, title = 'MSE Loss Heatmap'); plt.imshow(img_mse[i, settings.sel_slice, :, :], cmap = 'hot')
            plt.subplot(2, 2, 4, title = 'SSIM Index Mask'); plt.imshow(img_ssim[i, settings.sel_slice, :, :], cmap = plt.cm.binary)
        
            # Tensorboard Reconstruction Callback
            test_logger.experiment.add_figure(f"Target Results", test_plot, i)
            if i < settings.out_channels // 2: 
                test_logger.experiment.add_scalar(f"MSE Loss", mse_loss, i)
                test_logger.experiment.add_scalar(f"SSIM Index", ssim_loss, i)
            else:
                test_logger.experiment.add_scalar(f"MSE_Loss", mse_loss, i)
                test_logger.experiment.add_scalar(f"SSIM_Index", ssim_loss, i)
            

    # Result Image Saving
    img_fake = nib.Nifti1Image(img_fake.T, affine = np.eye(4)); img_fake.header.get_xyzt_units()
    img_mse = nib.Nifti1Image(img_mse.T, affine = np.eye(4)); img_mse.header.get_xyzt_units()
    img_ssim = nib.Nifti1Image(img_ssim.T, affine = np.eye(4)); img_ssim.header.get_xyzt_units()
    nib.save(img_fake, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/img_fake.nii.gz"))
    nib.save(img_mse, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/img_mse.nii.gz"))
    nib.save(img_ssim, Path(f"{checkpoint_folderpath}/test/epoch{save_epoch}/p{patient_id}/img_ssim.nii.gz"))
