# import packages
import time
import sys

import numpy as np
from multiprocessing import Pool
from scipy import interpolate


#===============================================================================================
# choose a ZTF time step
# temp = np.load("../SDSS_DR14_qso_mock_normal_sparse.npz", allow_pickle=True)
# ztf_time = temp["t_array"]
#ztf_time = temp["t_array"][164]
#choose_step = np.unique((ztf_time*10).astype("int"))
# print(choose_step.shape)
#
# # make a denser sampling
# for i in range(5):
#     choose_step = np.concatenate([choose_step,choose_step+3*i])
# choose_step = np.unique(choose_step)
# choose_step = choose_step[choose_step < 10000]
# print(choose_step.shape)

# load light curves
# temp = np.load("../SDSS_DR14_qso_mock_mixed_dense.npz")
temp = np.load("../Kelley_CAR1_mixed.npz")
t_array = temp["t_array"][:,:3000]
real_spec_all = temp["light_curve"][:,:3000]
print(real_spec_all.shape)

### change the amplitude
#real_spec_all = real_spec_all*10.


#================================================================================================
# load kernel
kernel = np.load("kernel_wavelet.npy")

# intiate array
Sx_all = []

#-----------------------------------------------------------------------------------------------
# calculate coefficient with uneven sampling
def calc_coefficient(j):

    print(j)

    # array collect coefficients
    Sx_all_temp = []

#-----------------------------------------------------------------------------------------------
    # choose time step
    # choose_step = np.unique((ztf_time[j]*10).astype("int"))
    #
    # # make a denser sampling
    # for i in range(10):
    #     choose_step = np.concatenate([choose_step,choose_step+i])
    # choose_step = np.unique(choose_step)
    # choose_step = choose_step[choose_step < 10000]
    # print(choose_step.shape)

#-----------------------------------------------------------------------------------------------
    # choose a spectrum
    real_spec = real_spec_all[j]
    time_stamp = t_array[j]
    # real_spec = real_spec_all[j][choose_step]
    # time_stamp = t_array[j][choose_step]

    # interpolate fascilitate convolution
    # f_power = interpolate.interp1d(choose_step, real_spec, kind='nearest',\
    #                            bounds_error=False, fill_value=(real_spec[0],real_spec[-1]))
    # real_spec = f_power(np.arange(10000))

#-------------------------------------------------------------------------------------
    # zero order coefficient
    real_spec_smooth = np.zeros(real_spec.size)
    real_spec_smooth = real_spec_smooth.astype("complex")

    # convolve with kernel
    for i in range(real_spec.size):
        real_spec_truncate = real_spec[np.max([0,i-4000]):i+4000+1]
        real_spec_smooth[i] = np.sum(real_spec_truncate*kernel[0][np.max([0,4000-i]):][:real_spec_truncate.size])
    Sx0 = np.median(np.absolute(real_spec_smooth))
    #Sx0 = np.median(np.absolute(real_spec_smooth[choose_step]))

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
        Sx_all_temp.append(np.median(np.absolute(real_spec_smooth)/Sx0))
        #Sx_all_temp.append(np.median(np.absolute(real_spec_smooth[choose_step])/Sx0))

#-------------------------------------------------------------------------------------
    # export results
    Sx_all_temp = np.array(Sx_all_temp)

    return Sx_all_temp

#-----------------------------------------------------------------------------------------------
# number of CPU to run in parallel
num_CPU = 8
pool = Pool(num_CPU)
Sx_all = np.array(pool.map(calc_coefficient,range(real_spec_all.shape[0])))
print(Sx_all.shape)

# save results
np.save("../Sx_all_mixed_dense.npy", Sx_all)
