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
class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim):
        super( ConvolutionalImplicitModel, self).__init__()
        self.z_dim = z_dim

        layers = []

        for i in range(5):
            for j in range(1):

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
                layers.append(torch.nn.LeakyReLU())

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

        # make it in 1D data image for DCI
        data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))

#-----------------------------------------------------------------------------------------------------------
        # draw random z
        z = torch.randn(batch_size*num_samples, self.z_dim, 1, 1).cuda()
        z_np_all = z.cpu().data.numpy()

        Sx = torch.from_numpy(np.repeat(data_Sx,num_samples_factor,axis=0)).float()[:z.shape[0]].cuda()
        Sx_np_all = Sx.cpu().data.numpy()

        z_Sx_all = torch.cat((z, Sx), axis=1)
        data_np_all = torch.from_numpy(data_np).float().cuda()

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
            # find the closest models
            if epoch % staleness == 0:
                samples_np = np.empty((num_samples*batch_size,)+data_np.shape[1:])

                # make in batch to avoid GPU memory problem
                for i in range(num_samples):
                    samples = self.model(z_Sx_all[i*batch_size:(i+1)*batch_size])
                    samples_np[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()

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

                # restrict to the nearest neighbour
                z_Sx = z_Sx_all[nearest_indices]


#=============================================================================================================
            # gradient descent
            err = 0.

            # save the mock sample
            if (epoch+1) % staleness == 0:
                samples_predict = np.empty(data_np.shape)

            # loop over all batches
            for i in range(num_batches):
                self.model.zero_grad()
                cur_samples = self.model(z_Sx[i*batch_size:(i+1)*batch_size])

                # save the mock sample
                if (epoch+1) % staleness == 0:
                    samples_predict[i*batch_size:(i+1)*batch_size] = cur_samples.cpu().data.numpy()

                # gradient descent
                loss = loss_fn(cur_samples, data_np_all[i*batch_size:(i+1)*batch_size])
                loss.backward()
                err += loss.item()
                optimizer.step()

            print("Epoch %d: Error: %f" % (epoch, err / num_batches))

#-----------------------------------------------------------------------------------------------------------
            # save the mock sample
            if (epoch+1) % staleness == 0:
                np.savez("../results_2D_j=1_clean.npz", data_np=data_np, Sx_np=Sx_np,\
                                samples_np=samples_predict)

                # make random mock
                samples_random = np.empty(data_np.shape)

                for i in range(num_batches):
                    samples = self.model(z_Sx_all[i*batch_size:(i+1)*batch_size])
                    samples_random[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()

                np.savez("../results_2D_random_j=1_clean.npz", samples_np=samples_random)


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
    torch.save(imle.model.state_dict(), '../net_weights_2D_j=1_clean.pth')

#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main(*sys.argv[1:])
