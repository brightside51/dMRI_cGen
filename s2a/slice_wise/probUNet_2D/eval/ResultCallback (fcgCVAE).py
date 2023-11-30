# Library Imports
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pytorch_lightning.loggers import TensorBoardLogger
from skimage.metrics import structural_similarity as ssim

# --------------------------------------------------------------------------------------------

# Result Plotting Functionality (V0)
def ResultCallback(
    settings,
    logger: TensorBoardLogger,
    best_info: dict,
    worst_info: dict,
    epoch: int = 0,
    mode: str = 'Train',
    loss: str = 'MSE Loss'
):

    # Set Example Reconstruction Results Image Plotting
    plot = plt.figure(figsize = (30, 25)); plt.suptitle(f"Overall {mode} | {loss}")
    plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
    
    # Best Loss Value Indexing
    best_loss = torch.empty(best_info['X_gt'].shape[0])
    for i in range(best_info['X_gt'].shape[0]):
        if loss == 'SSIM Index':
            ssim_loss, ssim_img = ssim( best_info['X_gt'][i, 0].cpu().numpy().astype(np.float32),
                                        best_info['X_fake'][i, 0].cpu().numpy().astype(np.float32), full = True,
                    data_range = (torch.max(best_info['X_gt']) - torch.min(best_info['X_gt'])).cpu().numpy())
            best_loss[i] = np.mean(ssim_loss); del ssim_img
        elif loss == 'MSE Loss':
            best_loss[i] = nn.MSELoss(reduction = 'mean')(  best_info['X_gt'][i],
                                                            best_info['X_fake'][i]).item()
        else: return logger
    best_idx = torch.argmax(best_loss)

    # Worst Loss Value Indexing
    worst_loss = torch.empty(worst_info['X_gt'].shape[0])
    for i in range(best_info['X_gt'].shape[0]):
        if loss == 'SSIM Index':
            ssim_loss, ssim_img = ssim( worst_info['X_gt'][i, 0].cpu().numpy().astype(np.float32),
                                        worst_info['X_fake'][i, 0].cpu().numpy().astype(np.float32), full = True,
                    data_range = (torch.max(worst_info['X_gt']) - torch.min(worst_info['X_gt'])).cpu().numpy())
            worst_loss[i] = np.mean(ssim_loss); del ssim_img
        elif loss == 'MSE Loss':
            worst_loss[i] = nn.MSELoss(reduction = 'mean')( best_info['X_gt'][i],
                                                            best_info['X_fake'][i]).item()
        else: return logger
    worst_idx = torch.argmin(worst_loss)

    # Set Example Original & Best Loss Reconstructed Image Subplots
    plt.subplot(2, 2, 1, title =    f"Best | Target Parameter #{best_info['idxh_target'][best_idx]}" +\
                                    f" | Target Slice #{best_info['idxv_slice'][best_idx]}")
    plt.imshow(best_info['X_gt'][   best_idx, 0, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 2, title =    f"Best | Reconstruction | {loss}: {np.round(best_loss[best_idx], 5)}")
    plt.imshow(best_info['X_fake'][ best_idx, 0, :, :], cmap = plt.cm.binary)

    # Set Example Original & Worst Loss Reconstructed Image Subplots
    plt.subplot(2, 2, 3, title =    f"Worst | Target Parameter #{worst_info['idxh_target'][worst_idx]}" +\
                                    f" | Target Slice #{worst_info['idxv_slice'][worst_idx]}")
    plt.imshow(worst_info['X_gt'][  worst_idx, 0, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 2, 4, title =    f"Worst | Reconstruction | {loss}: {np.round(worst_loss[worst_idx], 5)}")
    plt.imshow(worst_info['X_fake'][worst_idx, 0, :, :], cmap = plt.cm.binary)

    # Tensorboard Reconstruction Callback
    logger.experiment.add_figure(f"Image Results", plot, epoch)
    logger.experiment.add_scalar("Best Loss", best_loss[best_idx], epoch)
    logger.experiment.add_scalar("Worst Loss", worst_loss[worst_idx], epoch)
    return logger