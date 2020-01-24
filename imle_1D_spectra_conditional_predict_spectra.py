# In [0]:
# import packages
import numpy as np

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from astropy.io import fits

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#========================================================================================================
# define network
class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim):
        super( ConvolutionalImplicitModel, self).__init__()
        self.z_dim = z_dim
        layers = []
        channel = 256

        for i in range(6):
            for j in range(2):

                if i == 0 and j == 0:
                    layers.append(torch.nn.ConvTranspose1d(z_dim, channel, 7, stride=1))
                    layers.append(torch.nn.BatchNorm1d(channel, momentum=0.001, affine=False))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
                else:
                    layers.append(torch.nn.Conv1d(channel, channel, 5, stride=1, padding=2))
                    layers.append(torch.nn.BatchNorm1d(channel, momentum=0.001, affine=False))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))

            if i < 5:
                layers.append(torch.nn.Upsample(scale_factor=4, mode='linear', align_corners = False))
            else:
                layers.append(torch.nn.Conv1d(channel, 1, 6, stride=1))
                layers.append(torch.nn.LeakyReLU())

        self.model = torch.nn.Sequential(*layers)
        self.add_module("model", self.model)

    def forward(self, z):
        return self.model(z).view(-1,7163)


#========================================================================================================
# restore data
temp = np.load("../mock_all_spectra_no_noise_resample_prior_large.npz")
train_data = temp["spectra"][:,:7163]
train_Sx = temp["labels"].T
train_Sx[:,0] = train_Sx[:,0]/1000.

# shuffle the index
temp = np.load("../ind_shuffle_kurucz.npz")
ind_shuffle = temp["ind_shuffle"]
train_data = train_data[ind_shuffle,:][:12000,:]
train_Sx = train_Sx[ind_shuffle,:][:12000,:]

#-------------------------------------------------------------------------------------------------------
# restore models
Sx_dim = train_Sx.shape[1]
z_dim = 4
model = ConvolutionalImplicitModel(z_dim+Sx_dim).cuda()
state_dict = torch.load("../net_weights_spectra_deconv_256x2_epoch=2999.pth")
model.load_state_dict(state_dict)

# make predictions
Sx = torch.from_numpy(train_Sx).float().cuda()

#========================================================================================================
### predict with a single latent z ###
# z = torch.zeros(Sx.shape[0], z_dim).cuda()
# z_Sx_all = torch.cat((z, Sx), axis=1)[:,:,None]
#
# # train in batch
# batch_size = 100
# num_batches = z_Sx_all.shape[0] // batch_size
# predict_flux_array = []
# for i in range(num_batches):
#     print(i)
#     predict_flux_array.extend(model.forward(z_Sx_all[i*batch_size:(i+1)*batch_size]).cpu().data.numpy())
# predict_flux_array = np.array(predict_flux_array)


#========================================================================================================
### predict with random z and find the best estimates ###
predict_flux_array = []
num_samples_factor = 100
Sx = torch.from_numpy(np.repeat(train_Sx,num_samples_factor,axis=0)).float().cuda()
z = torch.randn(Sx.shape[0], z_dim).cuda()
z_Sx_all = torch.cat((z, Sx), axis=1)[:,:,None]

print(Sx.shape, z.shape)
print(Sx)
print(z_Sx_all.shape)

#for i in range(train_Sx.shape):
i = 0
predict_flux_temp = model.forward(z_Sx_all[i*num_sample_factor:(i+1)*num_sample_factor]).cpu().data.numpy()
print(predict_flux_temp.shape)
print(predict_flux_temp-train_data[i])

#========================================================================================================
# save array
np.save("../predict_flux_array.npy", predict_flux_array)
