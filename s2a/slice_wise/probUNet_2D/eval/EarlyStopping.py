# Library Imports
import pathlib
import numpy as np
import torch
from pathlib import Path
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# Early Stopping Callback Functionality
class EarlyStopping:

    # Constructor / Initialization Function
    def __init__(
        self,
        settings
    ):
        self.best_path = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Best 2D fcgCVAE.pt")
        self.local_path = Path(f"{settings.modelsave_folderpath}/V{settings.model_version}/Local 2D fcgCVAE.pt")
        self.patience = settings.es_patience; self.delta = settings.es_delta; self.early_stop = False
        self.counter = 0; self.best_score = -np.inf; self.local_score = -np.inf; self.local_loss = np.Inf; self.best_loss = np.inf
    
    # Callback Application Function
    def __call__(
        self,
        loss,
        model,
        optimizer,
        epoch: int = 0
    ):  

        # Best / First Local Result Checkpoint Saving
        if -loss > self.local_score + self.delta or self.local_score is None:
            self.save_checkpoint(loss, model, optimizer, epoch, best = False)
            self.local_score = -loss; self.counter = 0
        
        # Checkpoint Counter Startup
        else:
            self.counter += 1; self.local_score = -loss
            print(f'     > EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        
        # Best / First Overall Result Checkpoint Saving
        if -loss > self.best_score or self.best_score is None:
            self.save_checkpoint(loss, model, optimizer, epoch, best = True)
            self.best_score = -loss
        return self.early_stop

    # Model Saving Functionality
    def save_checkpoint(
        self,
        loss,
        model,
        optimizer,
        epoch: int = 0,
        best: bool = False
    ):
        if best: path = self.best_path
        else: path = self.local_path
        if best: print(f"     > Loss Decreased ({self.best_loss:.6f} --> {loss:.6f})\n     > Saving Best Model"); self.best_loss = loss
        else: print(f"     > Loss Decreased ({self.local_loss:.6f} --> {loss:.6f})\n     > Saving Local Model"); self.local_loss = loss
        torch.save({'Model': model.state_dict(),
                    'Optimizer': optimizer.state_dict(),
                    'Current Epoch': epoch,
                    'RNG State': torch.get_rng_state()}, path)