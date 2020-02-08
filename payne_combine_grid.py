import numpy as np

temp = np.load("../H3_training_grid_vt05.npz")
spectra_1 = temp["spectra"]
labels_1 = temp["labels"]
labels_1 = np.hstack([labels_1,np.array([np.ones(spectra_1.shape[0])*0.5]).T])
wavelength = temp["wavelength"]

temp = np.load("../H3_training_grid_vt10.npz")
spectra_2 = temp["spectra"]
labels_2 = temp["labels"]
labels_2 = np.hstack([labels_2,np.array([np.ones(spectra_2.shape[0])*1.0]).T])
wavelength = temp["wavelength"]

temp = np.load("../H3_training_grid_vt15.npz")
spectra_3 = temp["spectra"]
labels_3 = temp["labels"]
labels_3 = np.hstack([labels_3,np.array([np.ones(spectra_3.shape[0])*1.5]).T])
wavelength = temp["wavelength"]

temp = np.load("../H3_training_grid_vt20.npz")
spectra_4 = temp["spectra"]
labels_4 = temp["labels"]
labels_4 = np.hstack([labels_4,np.array([np.ones(spectra_4.shape[0])*2.0]).T])
wavelength = temp["wavelength"]

temp = np.load("../H3_training_grid_vt25.npz")
spectra_5 = temp["spectra"]
labels_5 = temp["labels"]
labels_5 = np.hstack([labels_5,np.array([np.ones(spectra_5.shape[0])*2.5]).T])
wavelength = temp["wavelength"]

temp = np.load("../H3_training_grid_vt30.npz")
spectra_6 = temp["spectra"]
labels_6 = temp["labels"]
labels_6 = np.hstack([labels_6,np.array([np.ones(spectra_6.shape[0])*3.0]).T])
wavelength = temp["wavelength"]

spectra = np.vstack([spectra_1,spectra_2,spectra_3,spectra_4,spectra_5,spectra_6])
labels = np.vstack([labels_1,labels_2,labels_3,labels_4,labels_5,labels_6])

print(spectra.shape)

np.savez("../H3_training_grid.npz",\
         spectra = spectra,
         labels = labels,
         wavelengt = wavelength)
