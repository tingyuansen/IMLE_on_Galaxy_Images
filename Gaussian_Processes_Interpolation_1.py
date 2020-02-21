# Gaussian Processes package
import GPy
import time
from multiprocessing import Pool
import os
import numpy as np

# set number of threads per CPU
os.environ['OMP_NUM_THREADS']='{:d}'.format(1)

#-------------------------------------------------------------------------------------
# restore grid
# temp = np.load("../SDSS_DR14_qso.npz", allow_pickle=True)
# mjd_g = temp["mjd_g"]
# g_array = temp["g_array"]

### restore mock grid ###
temp = np.load("../SDSS_DR14_qso_mock_normal_dense.npz", allow_pickle=True)
mjd_g_original = temp["t_array"]
mjd_g = temp["t_array"][:,::10]
g_array = temp["light_curve"][:,::10]

# set grid to interpolate into
# X_array = np.arange(5120)*0.1
X_array = mjd_g_original[0,:]
X_array = X_array.reshape(X_array.size,1)

#-------------------------------------------------------------------------------------
# interpolate with GP
def GP_interp(ind_choose):

    # extract a single light cure
    X = (mjd_g[ind_choose] - mjd_g[ind_choose][0])
    X = X.reshape(X.size,-1)
    Y = g_array[ind_choose]
    Y = Y.reshape(Y.size,-1)

    # define kernel
    k0 = GPy.kern.Matern32(1)
    k1 = GPy.kern.Matern32(1)
    k2 = GPy.kern.Matern32(1)

    kernel = k0
    #kernel = k0 + k1 + k2

#-------------------------------------------------------------------------------------
    # define regression
    m = GPy.models.GPRegression(X, Y, kernel, normalizer=True)

    # set range parameters
#     m.kern.Mat32.lengthscale.constrain_bounded(0.1,1)
#     m.kern.Mat32.variance.constrain_bounded(1e-10,1e-5)

#     m.kern.Mat32_1.lengthscale.constrain_bounded(1,10)
#     m.kern.Mat32_1.variance.constrain_bounded(1e-10,1e-5)

#     m.kern.Mat32_2.lengthscale.constrain_bounded(10,100)
#     m.kern.Mat32_2.variance.constrain_bounded(1e-10,1.)

    # fix the noise variance to known value
#     m.Gaussian_noise.variance = 1e-2**2
#     m.Gaussian_noise.variance.fix()

#-------------------------------------------------------------------------------------
    # optimize
    m.optimize(messages=True)
    m.optimize_restarts(num_restarts = 10)

    # make prediction
    Y_predict = np.array(m.predict(X_array))[0,:,0]

#-------------------------------------------------------------------------------------
    # # extract parameters
    # lengthscale_array = np.array([m.kern.Mat32.lengthscale[0],\
    #                               m.kern.Mat32_1.lengthscale[0],\
    #                               m.kern.Mat32_2.lengthscale[0]])
    # variance_array = np.array([m.kern.Mat32.variance[0],\
    #                            m.kern.Mat32_1.variance[0],\
    #                            m.kern.Mat32_2.variance[0]])
    #
    # # sort by lengthscale
    # length_sort = np.argsort(lengthscale_array)
    # lengthscale_array = lengthscale_array[length_sort]
    # variance_array = variance_array[length_sort]
    #
    # # combine all parameters
    # Y_predict = np.concatenate([lengthscale_array,variance_array])

#-------------------------------------------------------------------------------------
    ### assuming just one kernel
    # kernel = np.concatenate([m.kern.lengthscale,m.kern.variance])

#-------------------------------------------------------------------------------------
    # return prediction
    return Y_predict


#=====================================================================================
# number of CPU to run in parallel
num_CPU = 64
pool = Pool(num_CPU)
start_time = time.time()
Y_predict_array = np.array(pool.map(GP_interp,range(mjd_g.shape[0])))
print(time.time()-start_time)

# save results
np.save("../SDSS_DR14_qso_mock_normal_dense_GP_interpolated.npy", np.array(Y_predict_array))
