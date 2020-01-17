# import packages
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('../dci_code')
from dci import DCI


#=============================================================================================================
# define network
class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim):
        super(ConvolutionalImplicitModel, self).__init__()
        self.z_dim = z_dim
        self.features = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, 300),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(300, 7214),
        )

    def forward(self, x):
        return self.features(x)

#=============================================================================================================
# define class
class IMLE():
    def __init__(self, z_dim):
        self.z_dim = z_dim
        self.model = ConvolutionalImplicitModel(z_dim).cuda()
        self.dci_db = None

#-----------------------------------------------------------------------------------------------------------
    def train(self, data_np, base_lr=1e-4, batch_size=512, num_epochs=3000,\
             decay_step=25, decay_rate=0.95, staleness=100, num_samples_factor=100):

        # define metric
        loss_fn = nn.MSELoss().cuda()
        self.model.train()

        # train in batch
        num_batches = data_np.shape[0] // batch_size

        # truncate data to fit the batch size
        num_data = num_batches*batch_size
        data_np = data_np[:num_data]

#-----------------------------------------------------------------------------------------------------------
        # make empty array to store results
        samples_predict = np.empty(data_np.shape)
        samples_np = np.empty((num_samples_factor,)+data_np.shape[1:])

        # make global torch variables
        data_all = torch.from_numpy(data_np).float().cuda()

        # initiate dci
        if self.dci_db is None:
            self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices = 2, num_simp_indices = 7)


#=============================================================================================================
        # train through various epochs
        for epoch in range(num_epochs):

            # decay the learning rate
            if epoch % decay_step == 0:
                lr = base_lr * decay_rate ** (epoch // decay_step)
                optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

#-----------------------------------------------------------------------------------------------------------
            # update the closest models routintely
            if epoch % staleness == 0:

                # draw random z
                z_all = torch.randn(num_data*num_samples_factor, self.z_dim).cuda()

                # find the closest object for individual data
                nearest_indices = np.empty((num_data)).astype("int")

                for i in range(num_data):
                    samples = self.model(z_all[i*num_samples_factor:(i+1)*num_samples_factor])
                    samples_np[:] = samples.cpu().data.numpy()

#-----------------------------------------------------------------------------------------------------------
                    # find the nearest neighbours
                    self.dci_db.reset()
                    self.dci_db.add(np.copy(samples_np),\
                                    num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002)
                    nearest_indices_temp, _ = self.dci_db.query(data_np[i:i+1],\
                                        num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
                    nearest_indices[i] = nearest_indices_temp[0][0] + i*num_samples_factor

                # restrict latent parameters to the nearest neighbour
                z = z_all[nearest_indices]


#=============================================================================================================
            # gradient descent
            err = 0.

            # loop over all batches
            for i in range(num_batches):
                self.model.zero_grad()
                cur_samples = self.model(z[i*batch_size:(i+1)*batch_size])

                # save the mock sample
                if (epoch+1) % staleness == 0:
                    samples_predict[i*batch_size:(i+1)*batch_size] = cur_samples.cpu().data.numpy()

                # gradient descent
                loss = loss_fn(cur_samples, data_all[i*batch_size:(i+1)*batch_size])
                loss.backward()
                err += loss.item()
                optimizer.step()

            print("Epoch %d: Error: %f" % (epoch, err / num_batches))

#-----------------------------------------------------------------------------------------------------------
            # save the mock sample
            if (epoch+1) % staleness == 0:

                # save closet models
                np.savez("../results_spectra_" + str(epoch) +  ".npz", data_np=data_np,\
                                               z_np=z.cpu().data.numpy(),\
                                               samples_np=samples_predict)

                # save network
                torch.save(self.model.state_dict(), '../net_weights_spectra_epoch=' + str(epoch) + '.pth')


#=============================================================================================================
# run the codes
def main(*args):

    # restore data
    temp = np.load("../mock_all_spectra_no_noise_resample_prior_large.npz")
    train_data = temp["spectra"]
    print(train_data.shape)

#---------------------------------------------------------------------------------------------
    # initiate network
    z_dim = 50
    imle = IMLE(z_dim)

    # train the network
    imle.train(train_data)

#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main(*sys.argv[1:])
