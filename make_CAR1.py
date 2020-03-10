# import packages
import numpy as np
import time
from multiprocessing import Pool


#=============================================================================================================
# duration in day
duration = 1000

# define time array
t_clean = np.linspace(0, duration-0.1, int(duration*10.)) # cadence of 0.1 day
t_array = np.repeat(t_clean,1000).reshape(t_clean.size,1000).T


#=============================================================================================================
# make continuous autoregressive model 1
def make_CAR1(j):

    ### make the first component ###
    tau_1 = tau_array_1[j]
    sigma_1 = sigma_array_1[j]

    # the mean scale
    c_mag_1 = 17.

    # generate CAR1 covariance matrix
    r = np.fabs(t_clean[:,None] - t_clean[None,:])
    var = 0.5*tau_1*sigma_1**2
    cov = var*np.exp(-r/tau_1)

    # generate light curve
    n = len(t_clean)
    c_mag = c_mag_1*np.ones((n,)) # mean magnitude
    y_clean_1 = np.random.multivariate_normal(c_mag,cov)

    # return results
    return y_clean_1


#=============================================================================================================
# draw from distribution described in Kelley+ 09
tau_array_1 = 10**np.random.normal(2.75,0.66,1000)
sigma_array_1 = 10**np.random.normal(-2.04,0.23,1000)

#-----------------------------------------------------------------------------------------------
# number of CPU to run in parallel
start_time = time.time()
num_CPU = 4
pool = Pool(num_CPU)
light_curve = np.array(pool.map(make_CAR1,range(1000)))
print(light_curve.shape)
print(time.time()-start_time)

# save results
np.savez("../Kelley_CAR1_normal.npy",\
         light_curve = light_curve,\
         t_array = t_array,\
         tau_array = tau_array,\
         sigma_array = sigma_array)
