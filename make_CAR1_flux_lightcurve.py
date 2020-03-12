# import packages
import numpy as np
import time
from multiprocessing import Pool
import celerite
import emcee
from scipy.optimize import minimize


#=============================================================================================================
# make continuous autoregressive model 1
def make_CAR1(j):
    print(j)

    # duration
    duration = 1000

    # define time array
    t_clean = np.linspace(0, duration, int(duration))

#------------------------------------------------------------------------------------
    ### make the first component ###
    # choosing the mean from Kelley+ 09
    tau_1 = tau_1_array[j]
    sigma_1 = sigma_1_array[j]

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

    # convert into flux
    y_clean_1 = 10**((18-y_clean_1)/2.5)

    # add noise
    err = 0.01
    y_clean_1 = y_clean_1 + np.random.normal(0,err,len(y_clean_1))

#--------------------------------------------------------------------------------------------------------
    # define likelihood
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    # log posterior for emcee
    def log_post(params, y, gp):
        gp.set_parameter_vector(params)
        lp = gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(y)

#--------------------------------------------------------------------------------------------------------
    # MLE hyperparameter fit
    bounds = {"log_a": (-30,10), "log_c": (-20,0)}
    kernel = celerite.terms.RealTerm(log_a=4, log_c=-15, bounds=bounds)
    gp = celerite.GP(kernel, c_mag_1)
    gp.compute(t_clean, yerr=err)
    init = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    r = minimize(neg_log_like, init, method="L-BFGS-B", bounds=bounds, args=(y_clean_1, gp))

    # MCMC using emcee
    init = r.x
    ndim, nwalkers = len(init), 32
    init = r.x + np.random.normal(0, 1e-5, (nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args=(y_clean_1, gp))
    sampler.run_mcmc(init, 2000)
    samples = sampler.flatchain

#--------------------------------------------------------------------------------------------------------
    # prediction of the light curve
    gp.set_parameter_vector(np.median(samples, axis=0))
    yPred, varPred = gp.predict(y_clean_1, t_clean, return_var=True)
    stdPred = np.sqrt(varPred)

    # extract sample
    log_tau_sigma2 = samples[:,0]
    log_tau = -samples[:,1]
    log_sigma2 = log_tau_sigma2 - log_tau
    samples = np.vstack([log_tau,(log_sigma2+np.log(2))/2.]).T

    return np.median(samples,axis=0), np.std(samples,axis=0), stdPred


#=============================================================================================================
# CAR(1) parameters in magnitude space
tau_1_array = 10**np.random.normal(2.75,0.66,size=1000)
sigma_1_array = 10**np.random.normal(-2.04,0.23,size=1000)

# number of CPU to run in parallel
num_CPU = 96
pool = Pool(num_CPU)
results = pool.map(make_CAR1,range(1000))

# save results
np.savez("../Flux_CAR1",\
         results = results)
