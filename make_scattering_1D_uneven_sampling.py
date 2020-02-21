# import packages
import time
import sys

import numpy as np


#===============================================================================================
# load light curves
temp = np.load("../SDSS_DR14_qso_mock_normal_dense.npz")
t_array = temp["t_array"]
real_spec_all = temp["light_curve"]
print(real_spec_all.shape)

### change the amplitude
#real_spec = real_spec*10.

#================================================================================================
# choose windows of convolution (in unit of days)
window_array = 10.**np.linspace(-1,2,7)[::-1]


#================================================================================================
# intiate array
Sx_all = []

#-----------------------------------------------------------------------------------------------
# loop over all objects
#for j in range(real_spec.shape[0]):
for j in range(100):

    print(j)

    # array collect coefficients
    Sx_all_temp = []

    # choose a spectrum
    real_spec = real_spec_all[j,:]
    time_stamp = t_array[j,:]

    # normalize, i.e. substract away the zero order coefficients
    real_spec = real_spec/np.mean(np.abs(real_spec))

    # make smooth template
    real_spec_smooth = np.copy(real_spec)

#-----------------------------------------------------------------------------------------------
    # loop over all windows
    for k in range(window_array.shape[0]):

        # loop over all pixels
        for i in range(real_spec.size):

            # smooth pixel
            choose = np.abs(time_stamp - time_stamp[i]) < window_array[k]
            real_spec_smooth[i] = np.mean(real_spec[choose])

#-----------------------------------------------------------------------------------------------
        # substract away this frequency scale
        real_spec = real_spec - real_spec_smooth

        # extract coefficients
        Sx_all_temp.append(np.mean(np.abs(real_spec)))

    # record results
    Sx_all_temp = np.array(Sx_all_temp)
    Sx_all.append(Sx_all_temp)

#-----------------------------------------------------------------------------------------------
# save results
Sx_all = np.array(Sx_all)
print(Sx_all.shape)
np.save("../Sx_all_mixed_dense.npy", Sx_all)
