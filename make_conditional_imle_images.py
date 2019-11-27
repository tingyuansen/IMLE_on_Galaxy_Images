# import packages
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('../dci_code')
from dci import DCI


#=============================================================================================================
# # # define network
class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim, init_weight_factor = 1.):
        super( ConvolutionalImplicitModel, self).__init__()
        self.z_dim = z_dim
        self.init_weight_factor = init_weight_factor

        layers = []

        channel = 256

        for i in range(5):
            for j in range(2):

                if i == 0 and j == 0:
                    layers.append(torch.nn.ConvTranspose2d(z_dim, channel, 4, stride=1, padding=0))
                    layers.append(torch.nn.BatchNorm2d(channel, momentum=0.001, affine=False))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
                else:
                    layers.append(torch.nn.Conv2d(channel, channel, 5, stride=1, padding=2))
                    layers.append(torch.nn.BatchNorm2d(channel, momentum=0.001, affine=False))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))

            if i < 4:
                layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False))
            else:
                layers.append(torch.nn.Conv2d(channel, 1, 5, stride=1, padding=2))
                layers.append(torch.nn.LeakyReLU())

        self.model = torch.nn.Sequential(*layers)
        self.add_module("model", self.model)

    def forward(self, z):
        return self.model(z)

    def get_initializer(self):
        def initializer(m):
            if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                with torch.no_grad():
                    m.weight *= self.init_weight_factor
                    m.bias *= self.init_weight_factor
        return initializer


#=============================================================================================================
# define class
class IMLE():
    def __init__(self, z_dim, Sx_dim):
        self.z_dim = z_dim
        self.Sx_dim = Sx_dim
        self.model = ConvolutionalImplicitModel(z_dim+Sx_dim, 0.5).cuda()
        self.model.apply(self.model.get_initializer())
        self.dci_db = None

#-----------------------------------------------------------------------------------------------------------
        # load pre-trained model
        state_dict = torch.load("../net_weights_2D_conditional_times3_trial2_epoch=2999.pth")
        self.model.load_state_dict(state_dict)

#-----------------------------------------------------------------------------------------------------------
    def predict(self, data_np, data_Sx, batch_size=128, num_samples_factor=100):

        # define metric
        loss_fn = nn.MSELoss().cuda()
        self.model.train()

        # train in batch
        num_batches = data_np.shape[0] // batch_size

        # truncate data to fit the batch size
        num_data = num_batches*batch_size
        data_np = data_np[:num_data]
        data_Sx = data_Sx[:num_data]

        # repeat scattering scoefficients
        Sx = torch.from_numpy(np.repeat(data_Sx,num_samples_factor,axis=0)).float().cuda()

        # draw random z
        z = torch.randn(num_data*num_samples_factor, self.z_dim, 1, 1).cuda()
        z_Sx_all = torch.cat((z, Sx), axis=1)

        # make all different images of the same scattering coefficients
        for i in range(num_data):
            samples = self.model(z_Sx_all[i*num_samples_factor:(i+1)*num_samples_factor])
            np.savez("mock_images_" + str(i) + ".npz",\
                    data_np = data_np[i],\
                    samples_np = samples.cpu().data.numpy()


#=============================================================================================================
# run the codes
def main(*args):

    # restore data
    temp = np.load("../Illustris_Images.npz")
    train_data = temp["training_data"][::3,None,32:-32,32:-32]
    train_data = np.clip(np.arcsinh(train_data)+0.05,0,5)/5
    print(train_data.shape)

    # restore scattering coefficients
    train_Sx = np.load("../Sx_Illustris_Images.npy")[::3,:,None,None]
    print(train_Sx.shape)

#---------------------------------------------------------------------------------------------
    # initiate network
    z_dim = 4
    Sx_dim = train_Sx.shape[1]
    imle = IMLE(z_dim, Sx_dim)

    # train the network
    imle.predict(train_data, train_Sx)

#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main(*sys.argv[1:])
