# Library Imports
import torch
import torch.nn.functional as F
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)

# Functionality Import
from math import exp
torch.autograd.set_detect_anomaly(True)

# --------------------------------------------------------------------------------------------

# SSIM Loss Function
class SSIMLoss(torch.nn.Module):

    #https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

    # Constructor / Initialization Function
    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True
    ):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size; self.size_average = size_average
        self.num_channel = 1; self.window = self.create_window(window_size, self.num_channel)

    # Loss Application Function
    def forward(
        self,
        X_gt: torch.Tensor,
        X_pred: torch.Tensor
    ):

        # Loss Function Application
        num_channel = X_gt.shape[1]
        if num_channel == self.num_channel and self.window.data.type() == X_gt.data.type(): window = self.window
        else:
            window = self.create_window(self.window_size, num_channel)
            if X_gt.is_cuda: window = window.cuda(X_gt.get_device())
            window = window.type_as(X_gt)
            self.window = window; self.num_channel = num_channel
        return self.ssim_loss(X_gt, X_pred, window, self.window_size, num_channel, self.size_average)

    # --------------------------------------------------------------------------------------------

    # Gaussian Function Creation Functionality
    def gaussian(
        self,
        window_size: int,
        sigma: float
    ):
        gaussian = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gaussian / gaussian.sum()

    # Window Creation Functionality
    def create_window(
        self,
        window_size: int,
        num_channel: int
    ):
        window1D = self.gaussian(window_size, 1.5).unsqueeze(1)
        window2D = window1D.mm(window1D.t()).float().unsqueeze(0).unsqueeze(0)
        return Variable(window2D.expand(num_channel, 1, window_size, window_size).contiguous())

    # Structural Similarity Application Function
    def ssim_loss(
        self,
        X_gt: torch.Tensor,
        X_pred: torch.Tensor,
        window,
        window_size: int,
        num_channel: int,
        size_average: bool = True
    ):

        # Gaussian Properties Computation (Mu)
        mu1 = F.conv2d(X_gt, window, padding = window_size // 2, groups = num_channel)
        mu2 = F.conv2d(X_pred, window, padding = window_size // 2, groups = num_channel)
        square_mu1 = mu1.pow(2); square_mu2 = mu2.pow(2); C1 = 0.01**2; C2 = 0.03**2

        # Gaussian Properties Computation (Sigma)
        sigma1 = F.conv2d(X_gt * X_gt, window, padding = window_size // 2, groups = num_channel) - square_mu1
        sigma2 = F.conv2d(X_pred * X_pred, window, padding = window_size // 2, groups = num_channel) - square_mu2
        sigma = F.conv2d(X_gt * X_pred, window, padding = window_size//2, groups = num_channel) - (mu1 * mu2)

        # Structural Similarity Index Map Construction
        ssim_map = ((2 * (mu1 * mu2) + C1) * (2 * sigma + C2)) / ((square_mu1 + square_mu2 + C1) * (square_mu1 + square_mu2 + C2))
        if size_average: return ssim_map.mean()
        else: return ssim_map.mean(1).mean(1).mean(1)
