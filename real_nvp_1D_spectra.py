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
#temp = np.load("../mock_all_spectra_no_noise_resample_prior_large.npz")
#y_tr = temp["spectra"]

#-------------------------------------------------------------------------------------------------------
# temp = np.load("../apogee_spectra_0.npz")
# y_tr = temp["spectra"]
# y_tr[y_tr < 0.3] = 1.
# y_tr[y_tr > 1.5] = 1.

#-------------------------------------------------------------------------------------------------------
# read H3 labels
hdulist = fits.open('../H3_labels.fits')
SNR = hdulist[1].data['SNR']

# read H3 spectra
# H3 id, wave, flux, err, model, rest_wave
hdulist = fits.open('../H3_spectra.fits')
temp = hdulist[1].data

# define a uniform wavelength grid
uniform_wave = np.linspace(5162,5290,(5290-5162)*10.+1)

flux_spectra = np.empty((len(temp),uniform_wave.size))
for i in range(flux_spectra.shape[0]):
    if np.median(temp[i][2][0]) != 0:
        f_flux_spec = interpolate.interp1d(temp[i][5],temp[i][2])
        flux_spectra[i,:] = f_flux_spec(uniform_wave)
    else:
        flux_spectra[i,:] = temp[i][2][:uniform_wave.size]

# cull empty spectra
ind = np.where((np.median(flux_spectra, axis=1) != 0)*(SNR > 10.) == 1)[0]
flux_spectra = flux_spectra[ind,:]

# normalize spectra
y_tr = (flux_spectra.T/np.median(flux_spectra, axis=1)).T

# exclude bad pixels
y_tr[np.isnan(y_tr)] = 1.
y_tr[y_tr < 0.] = 0.
y_tr[y_tr > 2] = 2.
print(y_tr.shape)

#-------------------------------------------------------------------------------------------------------
# convert into torch
y_tr = torch.from_numpy(y_tr).type(torch.cuda.FloatTensor)


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



#=======================================================================================================
# In [3]:
# define network
device = torch.device("cuda")
num_neurons = 300

# input dimension
dim_in = y_tr.shape[-1]

nets = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in), nn.Tanh()).cuda()
nett = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in)).cuda()

# define mask
num_layers = 10
masks = []
for i in range(num_layers):
    mask_layer = np.random.randint(2,size=(dim_in))
    masks.append(mask_layer)
    masks.append(1-mask_layer)
masks = torch.from_numpy(np.array(masks).astype(np.float32))
masks.to(device)

# set prior
prior = distributions.MultivariateNormal(torch.zeros(dim_in, device='cuda'),\
                                         torch.eye(dim_in, device='cuda'))

# intiate flow
flow = RealNVP(nets, nett, masks, prior)
flow.cuda()


#=======================================================================================================
# In [4]
# number of epoch and batch size
num_epochs = 501
batch_size = 512

# break into batches
nsamples = y_tr.shape[0]
nbatches = nsamples // batch_size

# optimizing flow models
optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4)

#-------------------------------------------------------------------------------------------------------
# train the network
for e in range(num_epochs):

    # randomly permute the data
    perm = torch.randperm(nsamples)
    perm = perm.cuda()

    # For each batch, calculate the gradient with respect to the loss and take
    # one step.
    for i in range(nbatches):
        idx = perm[i * batch_size : (i+1) * batch_size]
        loss = -flow.log_prob(y_tr[idx]).mean()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    # the average loss.
    if e % 10 == 0:
        print('iter %s:' % e, 'loss = %.3f' % loss)


#========================================================================================================
# save models
torch.save(flow, 'flow_final.pt')

# sample results
z1 = np.empty(y_tr.shape)
for i in range(nbatches):
    z1[i*batch_size:(i+1)*batch_size] \
            = flow.f(y_tr[i*batch_size:(i+1)*batch_size])[0].detach().cpu().numpy()
z1_tr = torch.from_numpy(z1).type(torch.cuda.FloatTensor)
x1 = np.empty(y_tr.shape)
for i in range(nbatches):
    x1[i*batch_size:(i+1)*batch_size] \
    = flow.sample(z1_tr[i*batch_size:(i+1)*batch_size]).detach().cpu().numpy()


# save results
np.savez("../real_nvp_results.npz",\
         z1 = z1,\
         x1 = x1)
