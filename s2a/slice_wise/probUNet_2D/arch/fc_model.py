"""

FC model for voxel-wise DW-MR reconstruction.

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import torch
import torch.nn as nn
import numpy as np


class FCModel(nn.Module):

    def __init__(self,
                 in_features=505):

        super(FCModel, self).__init__()
        self.in_features = in_features

        self.net = nn.Sequential(nn.Linear(in_features=in_features,
                                           out_features=768),
                                 nn.BatchNorm1d(num_features=768),
                                 nn.ReLU(), # Input->H0

                                 nn.Linear(in_features=768,
                                           out_features=1024),
                                 nn.BatchNorm1d(num_features=1024),
                                 nn.ReLU(), # H0->H1

                                 nn.Linear(in_features=1024,
                                           out_features=768),
                                 nn.BatchNorm1d(num_features=768),
                                 nn.ReLU(), # H1->H2

                                 nn.Linear(in_features=768,
                                           out_features=512),
                                 nn.BatchNorm1d(num_features=512),
                                 nn.ReLU(), # H2->H3

                                 nn.Linear(in_features=512,
                                           out_features=1)) # H3->Output

    def forward(self, x):
        out = self.net(x)
        return out
