# import packages
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


#=============================================================================================================
# resnet models
class Payne_model(torch.nn.Module):
    def __init__(self, dim_in, num_neurons, num_features, mask_size, num_pixel):
        super(Payne_model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Linear(dim_in, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_neurons),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_neurons, num_features),
        )

        self.deconv1 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv2 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv3 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv4 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv5 = torch.nn.ConvTranspose1d(64, 64, mask_size, stride=3, padding=5)
        self.deconv6 = torch.nn.ConvTranspose1d(64, 32, mask_size, stride=3, padding=5)
        self.deconv7 = torch.nn.ConvTranspose1d(32, 1, mask_size, stride=3, padding=5)

        self.deconv2b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv3b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv4b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv5b = torch.nn.ConvTranspose1d(64, 64, 1, stride=3)
        self.deconv6b = torch.nn.ConvTranspose1d(64, 32, 1, stride=3)

        self.relu2 = torch.nn.LeakyReLU()
        self.relu3 = torch.nn.LeakyReLU()
        self.relu4 = torch.nn.LeakyReLU()
        self.relu5 = torch.nn.LeakyReLU()
        self.relu6 = torch.nn.LeakyReLU()

        self.num_pixel = num_pixel

    def forward(self, x):
        x = self.features(x)[:,None,:]
        x = x.view(x.shape[0], 64, 3)
        x1 = self.deconv1(x)

        x2 = self.deconv2(x1)
        x2 += self.deconv2b(x1)
        x2 = self.relu2(x2)

        x3 = self.deconv3(x2)
        x3 += self.deconv3b(x2)
        x3 = self.relu2(x3)

        x4 = self.deconv4(x3)
        x4 += self.deconv4b(x3)
        x4 = self.relu2(x4)

        x5 = self.deconv5(x4)
        x5 += self.deconv5b(x4)
        x5 = self.relu2(x5)

        x6 = self.deconv6(x5)
        x6 += self.deconv6b(x5)
        x6 = self.relu2(x6)

        x7 = self.deconv7(x6)[:,0,:self.num_pixel]
        return x7


#=============================================================================================================
# restore data
model = Payne_model(dim_in = 4, num_neurons = 300,\
                    num_features = 64*3, mask_size=11, num_pixel=4375)
state_dict = torch.load("../NN_normalized_spectra.pt")
model.load_state_dict(state_dict)

#------------------------------------------------------------------------------------------------
# reconstruct the xmin-xmax scaling
temp = np.load("../H3_training_grid.npz")
labels = temp["labels"]
labels[:,0] = labels[:,0]/1000.
x_max = np.max(labels, axis = 0)
x_min = np.min(labels, axis = 0)

#------------------------------------------------------------------------------------------------
# read spectra
temp = np.load("../H3_training_grid.npz")
labels = temp["labels"]
labels[:,0] = labels[:,0]/1000.
Y_u_all = temp["spectra"][:,150:4375+150].T

# calculate spectrum
labels = (labels[i,:]-x_min)/(x_max-x_min) - 0.5
predict_flux_array = model(torch.from_numpy(labels).float().cuda()).detach().cpu().numpy())

print(predict_flux_array.shape)
np.save("predict_flux_array.npy", predict_flux_array)
