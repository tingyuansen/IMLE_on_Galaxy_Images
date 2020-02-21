# import packages
import time
import sys

import numpy as np


#===============================================================================================
# load light curves
# temp = np.load("../SDSS_DR14_qso_mock_normal_dense.npz")
# real_spec = temp["light_curve"]
# print(real_spec.shape)

real_spec_all = np.load("../SDSS_DR14_qso_mock_mixed_dense_GP_interpolated.npy")
print(real_spec_all.shape)

### change the amplitude
#real_spec = real_spec*10.

## mix two modes
#real_spec = (real_spec[:,:] + real_spec[::-1,:])
#real_spec = -2.5*np.log10(10**(-real_spec[:,:]/2.5)+ 10**(-real_spec[::-1,:]/2.5))


#================================================================================================
# choose windows of convolution (in unit of days)
window_array = 10.**np.linspace(-1,2,7)[::-1]


#================================================================================================
# intiate array
Sx_all = []

#-----------------------------------------------------------------------------------------------
# loop over all objects
for i in range(real_spec.shape[0]):

    #
# make smooth template
real_spec_smooth = np.copy(real_spec)

# loop over all pixels
for i in range(real_spec.size):

    choose = np.abs(time_stamp - time_stamp[i]) < window_array[0]
    real_spec_smooth[i] = np.mean(real_spec[choose])
    real_spec_2_smooth[i] = np.mean(real_spec_2[choose])
    mixed_spec_smooth[i] = np.mean(mixed_spec[choose])

# substract away this frequency scale
real_spec = real_spec - real_spec_smooth
real_spec_2 = real_spec_2 - real_spec_2_smooth
mixed_spec = mixed_spec - mixed_spec_smooth

# calculate invariate representation
Sx_all = torch.mean(Sx_all, dim=-1)
Sx_all = Sx_all.cpu().numpy()

# normalize wrt to the first coefficient
for i in range(Sx_all.shape[0]):
    Sx_all[i,:] = Sx_all[i,:]/np.abs(Sx_all[i,0])

# take log to normalize the coefficient better
Sx_all = np.log10(Sx_all[:,1:])
print(Sx_all.shape)

# save results
np.save("../Sx_all_mixed_order=1_GP_interpolated.npy", Sx_all)
