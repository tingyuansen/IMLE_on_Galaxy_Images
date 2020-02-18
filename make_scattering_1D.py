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
real_spec = np.load("../light_curve.npy")
print(real_spec.shape)

### change the amplitude
#real_spec = real_spec*10.

## mix two modes
#real_spec = real_spec[:,:] + real_spec[::-1,:]


#================================================================================================
# define wavelet scattering hyperparameters
J = 6
Q = 8
T = real_spec.shape[1]

# convert into torch variable
x = torch.from_numpy(real_spec[:,:T]).type(torch.cuda.FloatTensor)
print(x.shape)

# define wavelet scattering
scattering = Scattering1D(J, T, Q)
scattering.cuda()

#================================================================================================
# perform wavelet scattering
Sx_all = scattering.forward(x)

# calculate invariate representation
Sx_all = torch.mean(Sx_all[:,1:,:], dim=-1)

# take log to normalize the coefficient better
Sx_all = torch.log10(Sx_all)
print(Sx_all.shape)

# save results
np.save("../Sx_all_normal.npy", Sx_all.cpu().numpy())
