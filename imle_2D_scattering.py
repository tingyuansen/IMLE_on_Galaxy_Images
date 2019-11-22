# import packages
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('./dci_code')
from dci import DCI


#=============================================================================================================
# define network
# class ConvolutionalImplicitModel(nn.Module):
#     def __init__(self, z_dim):
#         super(ConvolutionalImplicitModel, self).__init__()
#         self.z_dim = z_dim
#         self.tconv1 = nn.ConvTranspose2d(z_dim, 1024, 1, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(1024)
#         self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 3, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 3, padding=2, output_padding=1, bias=False)
#         self.bn4 = nn.BatchNorm2d(1)
#         self.relu = nn.LeakyReLU()
#
#     def forward(self, z):
#         z = self.relu(self.bn1(self.tconv1(z)))
#         z = self.relu(self.bn2(self.tconv2(z)))
#         z = self.relu(self.bn3(self.tconv3(z)))
#         z = self.relu(self.bn4(self.tconv4(z)))
#         return z

#-----------------------------------------------------------------------------------------------------------
# define network
class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim):
        super( ConvolutionalImplicitModel, self).__init__()
        self.z_dim = z_dim

        layers = []

        for i in range(5):
            for j in range(2):

                if i == 0 and j == 0:
                    layers.append(torch.nn.ConvTranspose2d(z_dim, 512, 4, stride=1, padding=0))
                    layers.append(torch.nn.BatchNorm2d(512, momentum=0.001, affine=False))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
                else:
                    layers.append(torch.nn.Conv2d(512, 512, 5, stride=1, padding=2))
                    layers.append(torch.nn.BatchNorm2d(512, momentum=0.001, affine=False))
                    layers.append(torch.nn.LeakyReLU(0.2, inplace=True))

            if i < 4:
                layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False))
            else:
                layers.append(torch.nn.Conv2d(512, 1, 5, stride=1, padding=2))
                layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)
        self.add_module("model", self.model)

    def forward(self, z):
        return self.model(z)


#=============================================================================================================
# define class
class IMLE():
    def __init__(self, z_dim, Sx_dim):
        self.z_dim = z_dim
        self.Sx_dim = Sx_dim
        self.model = ConvolutionalImplicitModel(z_dim+Sx_dim).cuda()
        self.dci_db = None

#-----------------------------------------------------------------------------------------------------------
    def train(self, data_np, data_Sx, base_lr=1e-3, batch_size=64, num_epochs=6000,\
              decay_step=25, decay_rate=1.0, staleness=100, num_samples_factor=10):

        # define metric
        loss_fn = nn.MSELoss().cuda()

        # make model trainable
        self.model.train()

        # train in batch
        num_batches = data_np.shape[0] // batch_size

        # true data to mock sample
        num_samples = num_batches * num_samples_factor

#-----------------------------------------------------------------------------------------------------------
        # make it in 1D data image for DCI
        data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))

#-----------------------------------------------------------------------------------------------------------
        # initiate dci
        if self.dci_db is None:
            self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices = 2, num_simp_indices = 7)

        # train through various epochs
        for epoch in range(num_epochs):

            # decay the learning rate
            if epoch % decay_step == 0:
                lr = base_lr * decay_rate ** (epoch // decay_step)
                optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

#-----------------------------------------------------------------------------------------------------------
            # re-evaluate the closest models routinely
            if epoch % staleness == 0:

                # initiate numpy array to store latent draws and the associate sample
                z_np = np.empty((num_samples*batch_size, self.z_dim, 1, 1))
                samples_np = np.empty((num_samples*batch_size,)+data_np.shape[1:])
                Sx_np = np.empty((num_samples*batch_size, self.Sx_dim, 1, 1))

                # draw random z
                #z = torch.randn(batch_size*num_samples, self.z_dim, 1, 1).cuda()

                # make sample (in batch to avoid GPU memory problem)
                for i in range(num_samples):

                    # draw random z
                    z = torch.randn(batch_size, self.z_dim, 1, 1).cuda()

                    # draw scattering coefficients from real data
                    ind_Sx = np.random.permutation(data_Sx.shape[0])
                    Sx = data_Sx[ind_Sx[:batch_size]]

#-----------------------------------------------------------------------------------------------------------
                    # predict sample
                    samples = self.model(torch.cat((z, torch.from_numpy(Sx).float().cuda()), axis=1))

                    # store the draws
                    z_np[i*batch_size:(i+1)*batch_size] = z.cpu().data.numpy()
                    samples_np[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()
                    Sx_np[i*batch_size:(i+1)*batch_size] = np.copy(Sx)

                # make 1D images
                samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], np.prod(samples_np.shape[1:])))

#-----------------------------------------------------------------------------------------------------------
                # find the nearest neighbours
                self.dci_db.reset()
                self.dci_db.add(np.copy(samples_flat_np),\
                                num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002)
                nearest_indices, _ = self.dci_db.query(data_flat_np,\
                                        num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
                nearest_indices = np.array(nearest_indices)[:,0]
                z_np = z_np[nearest_indices]
                Sx_np = Sx_np[nearest_indices]

                # add random noise to the latent space to faciliate training
                z_np += 0.01*np.random.randn(*z_np.shape)

                # delete to save Hyperparameters
                del samples_np, samples_flat_np


#=============================================================================================================
            # permute data
            #data_ordering = np.random.permutation(data_np.shape[0])
            #data_np = data_np[data_ordering]
            #data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))
            #z_np = z_np[data_ordering]
            #Sx_np = Sx_np[data_ordering]

#-----------------------------------------------------------------------------------------------------------
            # gradient descent
            err = 0.

            # save the mock sample
            if (epoch+1) % staleness == 0:
                samples_predict = np.empty(data_np.shape)

            # loop over all batches
            for i in range(num_batches):

                # set up backprop
                self.model.zero_grad()

#-----------------------------------------------------------------------------------------------------------
                # evaluate the models
                cur_z = torch.from_numpy(z_np[i*batch_size:(i+1)*batch_size]).float().cuda()
                cur_data = torch.from_numpy(data_np[i*batch_size:(i+1)*batch_size]).float().cuda()
                cur_Sx = torch.from_numpy(Sx_np[i*batch_size:(i+1)*batch_size]).float().cuda()
                cur_samples = self.model(torch.cat((cur_z,cur_Sx), axis=1))

                # save the mock sample
                if (epoch+1) % staleness == 0:
                    samples_predict[i*batch_size:(i+1)*batch_size] = cur_samples.cpu().data.numpy()

#-----------------------------------------------------------------------------------------------------------
                # calculate MSE loss of the two images
                loss = loss_fn(cur_samples, cur_data)
                loss.backward()
                err += loss.item()
                optimizer.step()

            print("Epoch %d: Error: %f" % (epoch, err / num_batches))

#-----------------------------------------------------------------------------------------------------------
            # save the mock sample
            if (epoch+1) % staleness == 0:
                np.savez("../results_2D_j=2.npz", data_np=data_np, Sx_np=Sx_np,\
                                samples_np=samples_predict)

                # make random mock
                samples_random = np.empty(data_np.shape)

                for i in range(num_batches):
                    z = torch.randn(batch_size, self.z_dim, 1, 1).cuda()
                    Sx = data_Sx[i*batch_size:(i+1)*batch_size]

                    # predict sample
                    samples = self.model(torch.cat((z, torch.from_numpy(Sx).float().cuda()), axis=1))
                    samples_random[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()

                np.savez("../results_2D_random_j=2.npz", samples_np=samples_random)


#=============================================================================================================
# run the codes
def main(*args):

    # restore data
    temp = np.load("../Illustris_Images.npz")
    train_data = temp["training_data"][:,None,32:-32,32:-32]
    train_data = np.clip(np.arcsinh(train_data)+0.05,0,5)/5
    print(train_data.shape)

    # restore scattering coefficients
    train_Sx = np.load("../Sx_Illustris_Images.npy")[:,:,None,None]
    print(train_Sx.shape)

#---------------------------------------------------------------------------------------------
    # initiate network
    z_dim = 32
    Sx_dim = train_Sx.shape[1]
    imle = IMLE(z_dim, Sx_dim)

    # train the network
    imle.train(train_data, train_Sx)
    torch.save(imle.model.state_dict(), '../net_weights_2D_j=2.pth')

#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main(*sys.argv[1:])
