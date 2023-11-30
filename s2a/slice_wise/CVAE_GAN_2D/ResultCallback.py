# Library Imports
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from pytorch_lightning.loggers import TensorBoardLogger

# --------------------------------------------------------------------------------------------

# Dataset Access
sys.path.append("../../../Datasets/MUDI Dataset/Dataset Reader")
from MUDI_CVAE_GAN_2D import MUDI_CVAE_GAN_2D

# --------------------------------------------------------------------------------------------

# Result Plotting Functionality (V0)
def ResultCallback(
    settings,
    logger: TensorBoardLogger,
    criterion,
    best_info: dict,
    worst_info: dict,
    epoch: int = 0,
    mode: str = 'Train',
    loss: str = 'MSE Loss',
):

    # Set Example Reconstruction Results Image Plotting
    plot = plt.figure(figsize = (30, 25)); plt.suptitle(f"Overall {mode} | {loss}")
    plt.xticks([]); plt.yticks([]); plt.grid(False); plt.tight_layout()
    
    # Source & GT Target Data Retrieval & Element-Wise Loss Computation
    best_info['X_train'], best_info['X_gt'] = MUDI_CVAE_GAN_2D.get_data(    settings, idxv = best_info['idxv_slice'],
                                                                            patient_id = best_info['patient_id'],
                                                                            idxh_source = best_info['idxh_source'],
                                                                            idxh_target = best_info['idxh_target'])
    worst_info['X_train'], worst_info['X_gt'] = MUDI_CVAE_GAN_2D.get_data(  settings, idxv = worst_info['idxv_slice'],
                                                                            patient_id = worst_info['patient_id'],
                                                                            idxh_source = worst_info['idxh_source'],
                                                                            idxh_target = worst_info['idxh_target'])
    
    # Best & Worst Loss Value Indexing
    if loss == 'SSIM Index':
        best_loss = criterion(  best_info['X_pred'][ np.arange(len(best_info['X_pred'])), np.arange(len(best_info['X_pred'])), :, :],
                                best_info['X_gt'][ np.arange(len(best_info['X_gt'])), np.arange(len(best_info['X_gt'])), :, :])
        worst_loss = criterion( worst_info['X_pred'][ np.arange(len(worst_info['X_pred'])), np.arange(len(worst_info['X_pred'])), :, :],
                                worst_info['X_gt'][ np.arange(len(worst_info['X_gt'])), np.arange(len(worst_info['X_gt'])), :, :])
    else:
        best_loss = torch.mean(criterion(best_info['X_pred'], best_info['X_gt']), dim = [1, 2, 3])
        worst_loss = torch.mean(criterion(worst_info['X_pred'], worst_info['X_gt']), dim = [1, 2, 3])
    best_auth_gt = best_info['auth_pred'][:len(best_info['auth_pred']) // 2] * 100
    best_auth_pred = best_info['auth_pred'][len(best_info['auth_pred']) // 2:] * 100
    worst_auth_gt = worst_info['auth_pred'][:len(worst_info['auth_pred']) // 2]
    worst_auth_pred = worst_info['auth_pred'][len(worst_info['auth_pred']) // 2:]
    best_idx = torch.argmax(best_loss); worst_idx = torch.argmin(worst_loss)

    # Set Example Original & Best Loss Reconstructed Image Subplots
    plt.subplot(2, 3, 1, title =    f"Best | Source Parameter #{best_info['idxh_source'][best_idx]}")
    plt.imshow(best_info['X_train'][best_idx, best_idx, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 2, title =    f"Best | Target Parameter #{best_info['idxh_target'][best_idx]}" +
                                    f" | Authenticity: {best_auth_gt[best_idx]}")
    plt.imshow(best_info['X_gt'][   best_idx, best_idx, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 3, title =    f"Best | Reconstruction | {loss}: {np.round(best_loss[best_idx], 5)}" +
                                    f" | Authenticity: {best_auth_pred[best_idx]}")
    plt.imshow(best_info['X_pred'][ best_idx, 0, :, :], cmap = plt.cm.binary)

    # Set Example Original & Worst Loss Reconstructed Image Subplots
    plt.subplot(2, 3, 4, title =    f"Best | Source Parameter #{worst_info['idxh_source'][worst_idx]}")
    plt.imshow(worst_info['X_train'][worst_idx, worst_idx, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 5, title =    f"Best | Target Parameter #{worst_info['idxh_target'][worst_idx]}" +
                                    f" | Authenticity: {worst_auth_gt[worst_idx]}")
    plt.imshow(worst_info['X_gt'][  worst_idx, worst_idx, :, :], cmap = plt.cm.binary)
    plt.subplot(2, 3, 6, title =    f"Best | Reconstruction | {loss}: {np.round(worst_loss[worst_idx], 5)}" +
                                    f" | Authenticity: {worst_auth_pred[worst_idx]}")
    plt.imshow(worst_info['X_pred'][worst_idx, 0, :, :], cmap = plt.cm.binary)

    # Tensorboard Reconstruction Callback
    logger.experiment.add_figure(f"Overall {mode} | {loss}", plot, epoch)
    logger.experiment.add_scalar(f"{loss} | Best ", best_loss[best_idx], epoch)
    logger.experiment.add_scalar(f"{loss} | Worst", worst_loss[worst_idx], epoch)
    logger.experiment.add_scalar(f"{loss} | Best | GT Authenticity", best_auth_gt[best_idx], epoch)
    logger.experiment.add_scalar(f"{loss} | Best | Gen Authenticity", best_auth_pred[best_idx], epoch)
    logger.experiment.add_scalar(f"{loss} | Worst | GT Authenticity", worst_auth_gt[worst_idx], epoch)
    logger.experiment.add_scalar(f"{loss} | Worst | Gen Authenticity", worst_auth_pred[worst_idx], epoch)
    return logger