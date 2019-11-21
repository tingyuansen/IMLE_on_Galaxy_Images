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
    train_data = temp["training_data"]

    # define scattering
    scattering = Scattering2D(J=5, shape=(training_x[0,:,:].shape), L=4, max_order=2)
    scattering.cuda()

    # transform to torch tensors
    tensor_training_x = torch.from_numpy(training_data).type(torch.cuda.FloatTensor)

    # perform scattering
    Sx = scattering(tensor_training_x).mean(dim=(2,3)).log().cpu().detach().numpy()

    # scattering coefficients
    np.save("Sx.npy", Sx)


#=========================================================================================================
if __name__ == '__main__':
    main()
