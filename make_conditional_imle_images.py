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
    def __init__(self, z_dim, Sx_dim, pix_choice):
        self.z_dim = z_dim
        self.Sx_dim = Sx_dim
        self.model = ConvolutionalImplicitModel(z_dim+Sx_dim, 0.5).cuda()
        self.model.apply(self.model.get_initializer())
        self.dci_db = None

#-----------------------------------------------------------------------------------------------------------
        # load pre-trained model
        state_dict = torch.load("../net_weights_2D_scattering_J=6_L=2_times=10.pth")
        # state_dict = torch.load("../net_weights_2D_times=10_inner_"+ str(pix_choice) \
        #                          + "x" + str(pix_choice) + ".pth")
        self.model.load_state_dict(state_dict)


#=============================================================================================================
    # make various realizations
    def predict(self, data_np, data_Sx, pix_choice, num_samples_factor=100):

        # initate result array
        num_data = data_Sx.shape[0]
        samples_np = np.empty((num_samples_factor*num_data,)+data_np.shape[1:])

        # repeat scattering scoefficients
        Sx = torch.from_numpy(np.repeat(data_Sx,num_samples_factor,axis=0)).float().cuda()

        # # draw random z
        z = torch.randn(Sx.shape[0], self.z_dim, 1, 1).cuda()
        z_Sx_all = torch.cat((z, Sx), axis=1)

        # make images in batch
        for i in range(num_samples_factor):
            samples_np[i::num_samples_factor] \
                    = self.model(z_Sx_all[i::num_samples_factor]).cpu().data.numpy()

        # save results
        np.savez("../samples_closest_" + str(pix_choice) \
                         + "x" + str(pix_choice) + ".npz",\
                    data_np = data_np,\
                    samples_np = samples_np)


#=============================================================================================================
    # # make various realizations
    # def predict(self, data_np, data_Sx, num_samples_factor=1000):
    #
    #     # initate result array
    #     samples_np = np.empty((num_samples_factor,)+data_np.shape[1:])
    #     num_base = data_Sx.shape[0]
    #
    #     # restore parameters
    #     temp = np.load("../results_2D_16x16_low_rez_times=10.npz")
    #     z_Sx_np = temp["z_Sx_np"][::30]
    #
    #     # repeat scattering scoefficients
    #     Sx_1 = torch.from_numpy(data_Sx).float().cuda()
    #
    #     # # draw random z
    #     z_1 = torch.randn(Sx_1.shape[0], self.z_dim, 1, 1).cuda()
    #
    #     data_Sx_add = np.linspace(z_Sx_np[45], z_Sx_np[55], num_samples_factor)
    #     z_Sx_2 = torch.from_numpy(data_Sx_add).float().cuda()
    #     z_Sx_1 = torch.cat((z_1, Sx_1), axis=1)
    #     z_Sx_all = torch.cat((z_Sx_1, z_Sx_2), axis=0)
    #
    #     # make images in batch
    #     for i in range(num_samples_factor):
    #         print(i)
    #         ind = np.concatenate([np.arange(num_base),np.array([i+num_base])])
    #         samples_np[i] = self.model(z_Sx_all[ind])[-1].cpu().data.numpy()
    #
    #     # save results
    #     np.savez("../samples_closest.npz",\
    #                 data_np = data_np,\
    #                 samples_np = samples_np)


#=============================================================================================================
# run the codes
def main(*args):

    # restore data
    temp = np.load("../Illustris_Images.npz")
    train_data = temp["training_data"][::30,None,32:-32,32:-32]
    train_data = np.clip(np.arcsinh(train_data)+0.05,0,5)/5
    print(train_data.shape)

#---------------------------------------------------------------------------------------------
    # restore scattering coefficients
    train_Sx = np.load("Sx_Illustris_Images_J=6_L=2.npy")[::30,:,None,None]
    print(train_Sx.shape)

    # # make low resolution as conditional
    pix_choice = int(args[0])

    # avg_choice = 64//pix_choice
    # train_Sx = np.empty((train_data.shape[0],)+(1,pix_choice,pix_choice))
    # for i in range(train_data.shape[0]):
    #     for j in range(pix_choice):
    #         for k in range(pix_choice):
    #             train_Sx[i,:,j,k] = np.mean(train_data[i,0,\
    #                                         j*avg_choice:(j+1)*avg_choice,\
    #                                         k*avg_choice:(k+1)*avg_choice])
    # train_Sx = train_Sx.reshape(train_Sx.shape[0],np.prod(train_Sx.shape[1:]),1,1)

    # train_Sx = np.empty((train_data.shape[0],)+(1,pix_choice*2,pix_choice*2))
    # for i in range(train_data.shape[0]):
    #     train_Sx[i,:,:,:] = train_data[i, 0 ,32-pix_choice:32+pix_choice, 32-pix_choice:32+pix_choice]
    # train_Sx = train_Sx.reshape(train_Sx.shape[0],np.prod(train_Sx.shape[1:]),1,1)

#---------------------------------------------------------------------------------------------
    # initiate network
    z_dim = 4
    Sx_dim = train_Sx.shape[1]
    imle = IMLE(z_dim, Sx_dim, pix_choice)

    # train the network
    imle.predict(train_data, train_Sx, pix_choice)

#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main(*sys.argv[1:])
