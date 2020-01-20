# In [0]:
# import packages
import numpy as np

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from astropy.io import fits


#========================================================================================================
# restore data
# temp = np.load("../mock_all_spectra_no_noise_resample_prior_large.npz")
# y_tr = temp["spectra"]

#-------------------------------------------------------------------------------------------------------
# temp = np.load("../apogee_spectra_1.npz")
# y_tr = temp["spectra"]
# y_tr[y_tr < 0.3] = 1.
# y_tr[y_tr > 1.5] = 1.

#-------------------------------------------------------------------------------------------------------
# H3 id, wave, flux, err, model, rest_wave
hdulist = fits.open('../H3_spectra.fits')
temp = hdulist[1].data

flux_spectra = np.empty((len(temp),temp[0][1].size))
model_spectra = np.empty((len(temp),temp[0][1].size))
for i in range(flux_spectra.shape[0]):
    flux_spectra[i,:] = temp[i][2]
    model_spectra[i,:] = temp[i][4]

# cull empty spectra
median_flux = np.median(flux_spectra, axis=1)
flux_spectra = flux_spectra[median_flux != 0,:]

# exclude pixels
flux_spectra = flux_spectra[:,100:-100]
y_tr = (flux_spectra.T/np.median(flux_spectra, axis=1)).T
y_tr[np.isnan(y_tr)] = 1.
y_tr[y_tr < 0.] = 0.
y_tr[y_tr > 2] = 2.
print(y_tr.shape)
print('yes')

#-------------------------------------------------------------------------------------------------------
# convert into torch
y_tr = torch.from_numpy(y_tr).type(torch.FloatTensor)

# input dimension
dim_in = y_tr.shape[-1]


#=======================================================================================================
# In [2]:
# define normalizing flow
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, z):
        x = self.g(z)
        return x


#==================================================================================
# restore models
flow = torch.load("../flow_final_h3.pt", map_location=lambda storage, loc: storage) # load in cpu
flow.eval()
print(flow.mask)

#-------------------------------------------------------------------------------------------------------
# sample results
# z1 = flow.f(y_tr)[0].detach().numpy()
# z1_tr = torch.from_numpy(z1).type(torch.FloatTensor)
# x1 = flow.sample(z1_tr).detach().numpy()
log_prob_x = flow.log_prob(y_tr).detach().numpy()
# print(log_prob_x.shape)

# train in batch
# batch_size = 1000
# num_batches = y_tr.shape[0] // batch_size
# log_prob_x = []
# for i in range(num_batches):
#     print(i)
#     log_prob_x.extend(flow.log_prob(torch.from_numpy(y_tr\
#                     [i*batch_size:(i+1)*batch_size]).type(torch.FloatTensor)).detach().numpy())
# print(log_prob_x.shape)

#-------------------------------------------------------------------------------------------------------
# save results
np.savez("../real_nvp_results_h3.npz",\
         #z1 = z1,\
         #x1 = x1,\
         log_prob_x = log_prob_x)
