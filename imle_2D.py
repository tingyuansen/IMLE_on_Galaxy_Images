# import packages
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.append('./dci_code')
from dci import DCI


#=============================================================================================================
# define network
class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim):
        super(ConvolutionalImplicitModel, self).__init__()
        self.z_dim = z_dim
        self.tconv1 = nn.ConvTranspose2d(z_dim, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 3, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tconv4 = nn.ConvTranspose2d(64, 3, 4, 3, padding=2, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(3)
        self.relu = nn.LeakyReLU()

    def forward(self, z):
        z = self.relu(self.bn1(self.tconv1(z)))
        z = self.relu(self.bn2(self.tconv2(z)))
        z = self.relu(self.bn3(self.tconv3(z)))
        z = self.relu(self.bn4(self.tconv4(z)))
        return z


#=============================================================================================================
# define class
class IMLE():
    def __init__(self, z_dim):
        self.z_dim = z_dim
        self.model = ConvolutionalImplicitModel(z_dim).cuda()
        self.dci_db = None

#-----------------------------------------------------------------------------------------------------------
    def train(self, data_np, base_lr=1e-3, batch_size=2048, num_epochs=30000,\
              decay_step=25, decay_rate=1.0, staleness=300, num_samples_factor=30):

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

        # initiate dci
        if self.dci_db is None:
            self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices = 2, num_simp_indices = 7)

#-----------------------------------------------------------------------------------------------------------
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

                # make sample (in batch to avoid GPU memory problem)
                for i in range(num_samples):
                    z = torch.randn(batch_size, self.z_dim, 1, 1).cuda()
                    samples = self.model(z)
                    z_np[i*batch_size:(i+1)*batch_size] = z.cpu().data.numpy()
                    samples_np[i*batch_size:(i+1)*batch_size] = samples.cpu().data.numpy()

                # make 1D images
                samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], np.prod(samples_np.shape[1:])))

#-----------------------------------------------------------------------------------------------------------
                # find the nearest neighbours
                self.dci_db.reset()
                self.dci_db.add(np.copy(samples_flat_np), num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002)
                nearest_indices, _ = self.dci_db.query(data_flat_np, num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
                nearest_indices = np.array(nearest_indices)[:,0]
                z_np = z_np[nearest_indices]

                # add random noise to the latent space to faciliate training
                z_np += 0.01*np.random.randn(*z_np.shape)

                # delete to save Hyperparameters
                del samples_np, samples_flat_np


#=============================================================================================================
            # permute data
            data_ordering = np.random.permutation(data_np.shape[0])
            data_np = data_np[data_ordering]
            data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))
            z_np = z_np[data_ordering]

#-----------------------------------------------------------------------------------------------------------
            # gradient descent
            err = 0.

            # loop over all batches
            for i in range(num_batches):

                # set up backprop
                self.model.zero_grad()

#-----------------------------------------------------------------------------------------------------------
                # evaluate the models
                cur_z = torch.from_numpy(z_np[i*batch_size:(i+1)*batch_size]).float().cuda()
                cur_data = torch.from_numpy(data_np[i*batch_size:(i+1)*batch_size]).float().cuda()
                cur_samples = self.model(cur_z)

#-----------------------------------------------------------------------------------------------------------
                # calculate MSE loss of the two images
                loss = loss_fn(cur_samples, cur_data)
                loss.backward()
                err += loss.item()
                optimizer.step()

            print("Epoch %d: Error: %f" % (epoch, err / num_batches))

            # save the mock sample
            if (epoch+1) % staleness == 0:
                np.savez("../results_2D_z_dim.npz", data_np=data_np,\
                        samples_np=self.model(torch.from_numpy(z_np).float().cuda()).cpu().data.numpy())


#=============================================================================================================
# run the codes
def main(*args):

    # restore data
    train_data = np.load("../training_set_des.npy")[::3]
    print(train_data.shape)

#---------------------------------------------------------------------------------------------
    # initiate network
    z_dim = 256
    imle = IMLE(z_dim)

    # train the network
    imle.train(train_data)
    torch.save(imle.model.state_dict(), '../net_weights_2D_z_dim.pth')

#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main(*sys.argv[1:])
