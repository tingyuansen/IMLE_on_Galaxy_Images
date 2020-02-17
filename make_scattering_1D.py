# import packages
from kymatio import Scattering1D
import kymatio

import torch.nn as nn
import torch.optim
import torch
import torch.utils.data as utils

import time
import sys

import numpy as np


#=========================================================================================================
# load light curves
light_curve = np.load("../light_curve.npy")

# define wavelet scattering hyperparameters
J = 12
Q = 5

# convert into torch variable
x = torch.from_numpy(light_curve).type(torch.cuda.FloatTensor)

# perform wavelet scattering
scattering = Scattering1D(J, light_curve.shape[1], Q)
scattering.cuda()
s = scattering.forward(x)[0]
print(s.shape)
