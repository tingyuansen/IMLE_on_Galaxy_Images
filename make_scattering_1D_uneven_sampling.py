# import packages
import time
import sys

import numpy as np
from multiprocessing import Pool


#===============================================================================================
# choose a ZTF time step
temp = np.load("../SDSS_DR14_qso_mock_normal_sparse.npz", allow_pickle=True)
ztf_time = temp["t_array"][133]
ztf_time = np.unique((ztf_time*10).astype("int"))
print(ztf_time.shape)

# load light curves
temp = np.load("../SDSS_DR14_qso_mock_validation_dense.npz")
t_array = temp["t_array"][:,ztf_time]
real_spec_all = temp["light_curve"][:,ztf_time]
print(real_spec_all.shape)

### change the amplitude
#real_spec_all = real_spec_all*10.
#real_spec_all = real_spec_all + 100.

# zero out the mean since WST is not addition invariant
#for i in range(real_spec_all.shape[0]):
#    real_spec_all[i,:] = real_spec_all[i,:] - np.mean(real_spec_all[i,:])


#================================================================================================
# choose windows of convolution (in unit of days)
window_array = 10.**np.linspace(-1,2,7)[::-1]


#================================================================================================
# intiate array
Sx_all = []

#-----------------------------------------------------------------------------------------------
# calculate coefficient with uneven sampling
def calc_coefficient(j):

    print(j)

    # array collect coefficients
    Sx_all_temp = []

    # choose a spectrum
    real_spec = real_spec_all[j]
    time_stamp = t_array[j]

    # make multiplicative invariant
    # by default it is additive invariant
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
np.save("../Sx_all_validation_dense.npy", Sx_all)
