# import packages
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

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
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.features(x)


#=============================================================================================================
# define class
class IMLE():
    def __init__(self, z_dim, Sx_dim):
        self.z_dim = z_dim
        self.Sx_dim = Sx_dim
        self.model = ConvolutionalImplicitModel(z_dim+Sx_dim).cuda()
        self.dci_db = None

#-----------------------------------------------------------------------------------------------------------
    def train(self, data_np, data_Sx, base_lr=1e-4, batch_size=512, num_epochs=10000,\
             decay_step=25, decay_rate=0.99, staleness=100, num_samples_factor=10):

        # define metric
        # loss_fn = nn.MSELoss().cuda()
        #loss_fn = nn.L1Loss().cuda()
        loss_fn = nn.BCELoss().cuda()

        self.model.train()

        # train in batch
        num_batches = data_np.shape[0] // batch_size

        # truncate data to fit the batch size
        num_data = num_batches*batch_size
        data_np = data_np[:num_data]
        data_Sx = data_Sx[:num_data]

#-----------------------------------------------------------------------------------------------------------
        # make empty array to store results
        samples_predict = np.empty(data_np.shape)

        #samples_np = np.empty((num_samples_factor,)+data_np.shape[1:])
        samples_np = np.empty((num_data*num_samples_factor,)+data_np.shape[1:])

        nearest_indices = np.empty((num_data)).astype("int")

        # make global torch variables
        data_all = torch.from_numpy(data_np).float().cuda()
        Sx = torch.from_numpy(np.repeat(data_Sx,num_samples_factor,axis=0)).float().cuda()

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
                z = torch.randn(num_data*num_samples_factor, self.z_dim).cuda()
                z_Sx_all = torch.cat((z, Sx), axis=1)

#-----------------------------------------------------------------------------------------------------------
                # # find the closest object for individual data
                # nearest_indices = np.empty((num_data)).astype("int")
                #
                # for i in range(num_data):
                #     samples = self.model(z_Sx_all[i*num_samples_factor:(i+1)*num_samples_factor])
                #     samples_np[:] = samples.cpu().data.numpy()
                #
                #     # find the nearest neighbours
                #     self.dci_db.reset()
                #     self.dci_db.add(np.copy(samples_np),\
                #                     num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002)
                #     nearest_indices_temp, _ = self.dci_db.query(data_np[i:i+1],\
                #                         num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
                #     nearest_indices[i] = nearest_indices_temp[0][0] + i*num_samples_factor

#-----------------------------------------------------------------------------------------------------------
                # find the closest object for individual data
                samples = self.model(z_Sx_all)
                samples_np[:] = samples.cpu().data.numpy()

                # find the nearest neighbours
                self.dci_db.reset()
                self.dci_db.add(np.copy(samples_np),\
                                num_levels = 2, field_of_view = 10, prop_to_retrieve = 0.002)
                nearest_indices_temp, _ = self.dci_db.query(data_np,\
                                num_neighbours = 1, field_of_view = 20, prop_to_retrieve = 0.02)
                nearest_indices[:] = nearest_indices_temp

#-----------------------------------------------------------------------------------------------------------
                # restrict latent parameters to the nearest neighbour
                z_Sx = z_Sx_all[nearest_indices]


#=============================================================================================================
            # gradient descent
            err = 0.

            # loop over all batches
            for i in range(num_batches):
                self.model.zero_grad()
                cur_samples = self.model(z_Sx[i*batch_size:(i+1)*batch_size])

                # save the mock sample
                if (epoch+1) % staleness == 0:
                    samples_predict[i*batch_size:(i+1)*batch_size] = cur_samples.cpu().data.numpy()

                # gradient descent
                np.savez("../test.npz",\
                         recon_x = cur_samples.cpu().data.numpy(),\
                         x = data_all[i*batch_size:(i+1)*batch_size].cpu().data.numpy())
                loss = loss_fn(cur_samples, data_all[i*batch_size:(i+1)*batch_size])
                loss.backward()
                err += loss.item()
                optimizer.step()

            print("Epoch %d: Error: %f" % (epoch, err / num_batches))

#-----------------------------------------------------------------------------------------------------------
            # save the mock sample
            if (epoch+1) % staleness == 0:

                # save closet models
                np.savez("../results_spectra_BCE_" + str(epoch) +  ".npz", data_np=data_np,\
                                               z_Sx_np=z_Sx.cpu().data.numpy(),\
                                               samples_np=samples_predict)

                np.savez("../mse_err_BCE_" + str(epoch) +  ".npz",\
                                                mse_err=err/num_batches)

                # save network
                torch.save(self.model.state_dict(), '../net_weights_spectra_BCE_epoch=' + str(epoch) + '.pth')


#=============================================================================================================
# run the codes
def main(*args):

    # restore data
    temp = np.load("../mock_all_spectra_no_noise_resample_prior_large.npz")
    train_data = temp["spectra"]/2.
    train_Sx = temp["labels"].T
    train_Sx[:,0] = train_Sx[:,0]/1000.
    print(train_data.shape)
    print(train_Sx.shape)

#---------------------------------------------------------------------------------------------
    # shuffle the index
    # ind_shuffle = np.arange(train_data.shape[0])
    # np.random.shuffle(ind_shuffle)
    # train_data = train_data[ind_shuffle,:][:12000,:]
    # train_Sx = train_Sx[ind_shuffle,:][:12000,:]
    # np.savez("../ind_shuffle.npz", ind_shuffle=ind_shuffle)

#---------------------------------------------------------------------------------------------
    # initiate network
    z_dim = 4
    Sx_dim = train_Sx.shape[1]
    imle = IMLE(z_dim, Sx_dim)

    # train the network
    imle.train(train_data, train_Sx)

#---------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main(*sys.argv[1:])
