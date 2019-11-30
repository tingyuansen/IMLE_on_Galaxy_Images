# import packages
from kymatio import Scattering2D
import kymatio

import torch.nn as nn
import torch.optim
import torch
import torch.utils.data as utils

import time
import sys

import numpy as np


#=========================================================================================================
# main body of the script
def main():

    # load data
    temp = np.load("../Illustris_Images.npz")
    training_x = temp["training_data"]

    # define scattering
    scattering = Scattering2D(J=6, shape=(training_x[0,:,:].shape), L=4, max_order=2)
    scattering.cuda()

    # initiate results array
    Sx = []

#----------------------------------------------------------------------------------------------------------
    # loop over batches of 500 objects
    for i in range(training_x.shape[0]//100+1):
        print(i)

        # record time
        start_time = time.time()

        # transform to torch tensors
        tensor_training_x = torch.from_numpy(training_x[100*i:100*(i+1),:,:]).type(torch.cuda.FloatTensor)

        # perform scattering
        Sx.append(scattering(tensor_training_x).mean(dim=(2,3)).log().cpu().detach().numpy())
        print(time.time() - start_time)

#----------------------------------------------------------------------------------------------------------
    # save results
    for i in range(len(Sx)):
        try:
            Sx_array = np.vstack([Sx_array,Sx[i]])
        except:
            Sx_array = Sx[i]
        print(Sx_array.shape)
    np.save("../Sx_Illustris_Images_J=6_L=2.npy", Sx_array)


#=========================================================================================================
if __name__ == '__main__':
    main()
