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
        state_dict = torch.load("../net_weights_2D_conditional_times3_repeat_no_decay_epoch=499.pth")
        self.model.load_state_dict(state_dict)

#-----------------------------------------------------------------------------------------------------------
    def predict(self, data_np, data_Sx, num_samples_factor=1):

        # initate result array
        num_data = data_Sx.shape[0]
        samples_np = np.empty((num_samples_factor*num_data,)+data_np.shape[1:])

        # repeat scattering scoefficients
        Sx = torch.from_numpy(np.repeat(data_Sx,num_samples_factor,axis=0)).float().cuda()

        # # draw random z
        z = torch.randn(Sx.shape[0], self.z_dim, 1, 1).cuda()
        z_Sx_all = torch.cat((z, Sx), axis=1)

        # make images in batch
        #for i in range(num_samples_factor):
        #    samples_np[i*num_data:(i+1)*num_data] \
        #            = self.model(z_Sx_all[i*num_data:(i+1)*num_data]).cpu().data.numpy()

        # save results
        np.savez("../sample_closest.npz",\
                    data_np = data_np,\
                    samples_np = self.model(z_Sx_all).cpu().data.numpy())


#=============================================================================================================
#         # make it in 1D data image for DCI
#         data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))
#
#         # make empty array to store results
#         samples_predict = np.empty(data_np.shape)
#         samples_np = np.empty((num_samples_factor,)+data_np.shape[1:])
#
#         # initiate dci
#         if self.dci_db is None:
#             self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices = 2, num_simp_indices = 7)
#
#         # draw random z
#         z = torch.randn(num_data*num_samples_factor, self.z_dim, 1, 1).cuda()
#         z_Sx_all = torch.cat((z, Sx), axis=1)
#
# #-----------------------------------------------------------------------------------------------------------
#         # find the closest object for individual data
#         nearest_indices = np.empty((num_data)).astype("int")
#
#         # find the cloest models
#         for i in range(num_data):
#             samples = self.model(z_Sx_all[i*num_samples_factor:(i+1)*num_samples_factor])
#             samples_np[:] = samples.cpu().data.numpy()
#             samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], np.prod(samples_np.shape[1:])))
#
#             # find the nearest neighbours
#             self.dci_db.reset()
#             self.dci_db.add(np.copy(samples_flat_np),\
#                                     num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002)
#             nearest_indices_temp, _ = self.dci_db.query(data_flat_np[i:i+1],\
#                                     num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
#             nearest_indices[i] = nearest_indices_temp[0][0] + i*num_samples_factor
#
# #-----------------------------------------------------------------------------------------------------------
#         # restrict latent parameters to the nearest neighbour
#         z_Sx = z_Sx_all[nearest_indices]
#
#         # loop over all batches
#         for i in range(num_batches):
#             cur_samples = self.model(z_Sx[i*batch_size:(i+1)*batch_size])
#             samples_predict[i*batch_size:(i+1)*batch_size] = cur_samples.cpu().data.numpy()
#
#         # save results
#         np.savez("../samples_closest.npz",\
#                  data_np=data_np, samples_np=samples_predict)


#=============================================================================================================
# run the codes
def main(*args):

    # restore data
    temp = np.load("../Illustris_Images.npz")
    train_data = temp["training_data"][::100,None,32:-32,32:-32]
    train_data = np.clip(np.arcsinh(train_data)+0.05,0,5)/5
    print(train_data.shape)

    # restore scattering coefficients
    train_Sx = np.load("../Sx_Illustris_Images.npy")[::100,:,None,None]
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
