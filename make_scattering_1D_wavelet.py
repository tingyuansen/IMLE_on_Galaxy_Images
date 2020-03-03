# import packages
import time
import sys

import numpy as np
from multiprocessing import Pool


#===============================================================================================
# load light curves
temp = np.load("../SDSS_DR14_qso_mock_validation_dense.npz")
real_spec_all = temp["light_curve"]
print(real_spec_all.shape)

### change the amplitude
#real_spec_all = real_spec_all*10.

# load kernel
kernel = np.load("kernel_wavelet.npy")


#================================================================================================
# intiate array
Sx_all = []

#-----------------------------------------------------------------------------------------------
# calculate coefficient with uneven sampling
def calc_coefficient(j):

    print(j)

    # array collect coefficients
    Sx_all_temp = []

#-----------------------------------------------------------------------------------------------
    # choose a spectrum
    real_spec = real_spec_all[j]
    #time_stamp = t_array[j]

#-------------------------------------------------------------------------------------
    # zero order coefficient
    real_spec_smooth = np.zeros(real_spec.size)
    real_spec_smooth = real_spec_smooth.astype("complex")

    # convolve with kernel
    for i in range(real_spec.size):
        real_spec_truncate = real_spec[np.max([0,i-4000]):i+4000+1]
        real_spec_smooth[i] = np.sum(real_spec_truncate*kernel[0][np.max([0,4000-i]):][:real_spec_truncate.size])
    Sx0 = np.median(np.absolute(real_spec_smooth))

#-------------------------------------------------------------------------------------
    # make smooth template
    real_spec_smooth = np.copy(real_spec)

    # loop over all windows
    for k in range(len(kernel)-1):

        # iniatite array
        real_spec_smooth = np.zeros(real_spec.size)
        real_spec_smooth = real_spec_smooth.astype("complex")

        # convolve with kernel
        for i in range(real_spec.size):
            real_spec_truncate = real_spec[np.max([0,i-4000]):i+4000+1]
            real_spec_smooth[i] = np.sum(real_spec_truncate*kernel[k+1][np.max([0,4000-i]):][:real_spec_truncate.size])

        # extract coefficients
        Sx_all_temp.append(np.median(np.absolute(real_spec_smooth[100:-100])/Sx0))

#-------------------------------------------------------------------------------------
    # export results
    Sx_all_temp = np.array(Sx_all_temp)

    return Sx_all_temp

#-----------------------------------------------------------------------------------------------
# number of CPU to run in parallel
num_CPU = 64
pool = Pool(num_CPU)
Sx_all = np.array(pool.map(calc_coefficient,range(real_spec_all.shape[0])))
print(Sx_all.shape)

# save results
np.save("../Sx_all_validation_dense.npy", Sx_all)
