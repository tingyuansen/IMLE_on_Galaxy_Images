# import packages
import numpy as np
from The_Payne import training


#==========================================================================================
# read spectra
temp = np.load("../Kurucz_CK3_Optical_R=45000_MIKE.npz")
spectra = temp["spectra"]
labels = temp["labels"]
labels[:,0] = labels[:,0]/1000.

print(labels.shape)

#---------------------------------------------------------------------------------------
# reshuffle the array
ind_shuffle = np.arange(spectra.shape[0])
np.random.shuffle(ind_shuffle)
np.savez("../ind_shuffle_payne_h3.npz", ind_shuffle=ind_shuffle)

#----------------------------------------------------------------------------------------
# separate into training and validation set
spectra = spectra[ind_shuffle,:]
labels = labels[ind_shuffle,:]
training_spectra = spectra[:18000,:]
training_labels = labels[:18000,:]
validation_spectra = spectra[18000:,:]
validation_labels = labels[18000:,:]
print(training_spectra.shape)
print(validation_spectra.shape)

#----------------------------------------------------------------------------------------
# train neural network # require GPU
training_loss, validation_loss = training.neural_net(training_labels, training_spectra,\
                                                     validation_labels, validation_spectra,\
                                                     num_neurons=1000, learning_rate=1e-4,\
                                                     num_steps=1e7, batch_size=32, num_pixel=spectra.shape[1])
