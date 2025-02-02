# import packages
import numpy as np
import time
from multiprocessing import Pool


#=============================================================================================================
# duration in day
duration = 300

# define time array
t_clean = np.linspace(0, duration-0.1, int(duration*10)) # cadence of 0.1 day
t_array = np.repeat(t_clean,1000).reshape(t_clean.size,1000).T
r = np.fabs(t_clean[:,None] - t_clean[None,:])
n = len(t_clean)


#=============================================================================================================
# make continuous autoregressive model 1
def make_CAR1(j):
    print(j)

    ### make the first component ###
    tau_1 = tau_array_1[j]
    sigma_1 = sigma_array_1[j]

    # the mean scale
    c_mag_1 = 17.
    c_mag = c_mag_1*np.ones((n,)) # mean magnitude

    # generate CAR1 covariance matrix
    var = 0.5*tau_1*sigma_1**2
    cov = var*np.exp(-r/tau_1)

    # generate light curve
    y_clean_1 = np.random.multivariate_normal(c_mag,cov)
    return y_clean_1



#=============================================================================================================
# draw from distribution described in Kelley+ 09
#tau_array_1 = 10**np.random.normal(2.75,0.66,1000)
#sigma_array_1 = 10**np.random.normal(-2.04,0.23,1000)
tau_array_1 = 10**np.random.normal(2.75,0.06,1000)
sigma_array_1 = 10**np.random.normal(-2.04,0.02,1000)

#-----------------------------------------------------------------------------------------------
# number of CPU to run in parallel
start_time = time.time()
num_CPU = 96
pool = Pool(num_CPU)
light_curve = np.array(pool.map(make_CAR1,range(1000)))
print(light_curve.shape)
print(time.time()-start_time)

#-----------------------------------------------------------------------------------------------
# save results
np.savez("../Kelley_CAR1_validation",\
         light_curve = light_curve,\
         t_array = t_array,\
         tau_array = tau_array_1,\
         sigma_array = sigma_array_1)
# np.savez("../Kelley_CAR1_mixed",\
#          light_curve = light_curve,\
#          t_array = t_array,\
#          tau_array_1 = tau_array_1,\
#          sigma_array_1 = sigma_array_1,\
#          tau_array_2 = tau_array_2,\
#          sigma_array_2 = sigma_array_2)
