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
    
    # Set Example Original & Best Loss Reconstructed Image Subplots
    plt.subplot(2, 2, 1, title =    f"Best | Patient #{best_info['patient_id']} | " +\
                                    f"Target Parameter #{best_info['idxh_target']} | " +\
                                    f"Target Slice #{best_info['idxv_slice']}")
    plt.imshow(best_info['X_gt'], cmap = plt.cm.binary)
    plt.subplot(2, 2, 2, title =    f"Best | Reconstruction | {loss}: {np.round(best_info['loss'], 5)}")
    plt.imshow(best_info['X_fake'], cmap = plt.cm.binary)

    # Set Example Original & Worst Loss Reconstructed Image Subplots
    plt.subplot(2, 2, 3, title =    f"Worst | Patient #{worst_info['patient_id']} | " +\
                                    f"Target Parameter #{worst_info['idxh_target']} | " +\
                                    f"Target Slice #{worst_info['idxv_slice']}")
    plt.imshow(worst_info['X_gt'], cmap = plt.cm.binary)
    plt.subplot(2, 2, 4, title =    f"Worst | Reconstruction | {loss}: {np.round(worst_info['loss'], 5)}")
    plt.imshow(worst_info['X_fake'], cmap = plt.cm.binary)

    # Tensorboard Reconstruction Callback
    logger.experiment.add_figure(f"Image Results", plot, epoch)
    logger.experiment.add_scalar("Best Loss", best_info['loss'], epoch)
    logger.experiment.add_scalar("Worst Loss", worst_info['loss'], epoch)
    return logger