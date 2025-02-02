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


#===============================================================================================
# load light curves
temp = np.load("../SDSS_DR14_qso_mock_normal_dense.npz")
real_spec = temp["light_curve"]
print(real_spec.shape)

# real_spec = np.load("../SDSS_DR14_qso_mock_mixed_dense_GP_interpolated.npy")
# print(real_spec.shape)

### change the amplitude
#real_spec = real_spec*10.
real_spec = real_spec + 100.


#================================================================================================
# define wavelet scattering hyperparameters
J = 10
Q = 1
T = real_spec.shape[1]
max_choice = 1

# convert into torch variable
x = torch.from_numpy(real_spec[:,:T]).type(torch.cuda.FloatTensor)
print(x.shape)

# define wavelet scattering
scattering = Scattering1D(J, T, Q, max_order=max_choice)
scattering.cuda()


#================================================================================================
# perform wavelet scattering
Sx_all = scattering.forward(x)

# calculate invariate representation
Sx_all = torch.mean(Sx_all.abs(), dim=-1)
Sx_all = Sx_all.cpu().numpy()

# make multiplicative invariant
#for i in range(Sx_all.shape[0]):
#    Sx_all[i,:] = Sx_all[i,:]/Sx_all[i,0]

# the default is additive invariant

# save results
np.save("../Sx_all_normal_dense_add100.npy", Sx_all[:,1:])
