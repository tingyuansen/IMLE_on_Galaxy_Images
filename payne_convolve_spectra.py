# import packages
import os
import numpy as np
import h5py
from phil_h3_smoothing import smoothspec


#===============================================================================
# all the hdf5 files
vt_str = sys.argv[1]
file_list = os.listdir("/n/conroyfs1/bdjohnson/data/stars/c3k_v1.3/rv31_" + vt_str)


# loop over all files
spectra = []
parameters = []
wavelength = []
for file_name in file_list:
    fcat = h5py.File('/n/conroyfs1/bdjohnson/data/stars/c3k_v1.3/rv31_vt' + vt_str + '/' + file_name, 'r')
    spectra.extend(np.array(fcat['spectra'])/np.array(fcat['continuua']))
    wavelength = np.array(fcat['wavelengths'])
    parameters_temp = np.array(fcat['parameters'])
    parameters.extend(np.array([np.array(list(parameters_temp[i])) \
                                for i in range(len(parameters_temp))]))
spectra = np.array(spectra)
parameters = np.array(parameters)
wavelength = np.array(wavelength)
print(spectra.shape)
print(parameters.shape)
print(wavelength.shape)

#---------------------------------------------------------------------------------
# convert log Teff to Teff
parameters[:,0] = 10.**parameters[:,0]

# restrict to Teff = 3500-10000, logg = 0-5
ind = (parameters[:,0] >= 3500.)*(parameters[:,0] <= 10000.)\
       *(parameters[:,1] >= 0.)*(parameters[:,1] <= 5.)
parameters = parameters[ind,:]
spectra = spectra[ind,:]


#===============================================================================
# define wavelength range
waverange = [5100, 5350]

# define output wavleength
wavelength_i = np.copy(wavelength)
wavelength_o = []
resolution_o = 100000
resolution = resolution_o/2.355

i = 1
while True:
    wave_i = waverange[0]*(1.0 + 1.0/(3.0*resolution))**(i-1.0)
    if wave_i <= waverange[1]:
        wavelength_o.append(wave_i)
        i += 1
    else:
        break
wavelength_o = np.array(wavelength_o)

# smooth the spectra
spectra_o = []
for i in range(spectra.shape[0]):
    spectra_o.append(smoothspec(wavelength_i,spectra[i,:], resolution_o,\
                                outwave=wavelength_o, smoothtype='R', fftsmooth=True, inres=500000.0))
spectra = np.copy(spectra_o)

np.savez("../H3_training_grid_vt" + vt_str + ".npz",\
         labels = parameters,\
         spectra = spectra,\
         wavelength = wavelength_o)
print(parameters.shape)
print(spectra.shape)
print(wavelength.shape)
