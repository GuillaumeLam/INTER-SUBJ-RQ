# script for py_compatible_data.mat => X.npy
import scipy.io
import numpy as np

data = scipy.io.loadmat('../py_compatible_data.mat')
gait_ireg_surface_dataset = data['gait_ireg_surface_dataset']
gait_ireg_surface_dataset = np.array([[e[0][0][0], e[1], e[2][0][0][0]] for e in gait_ireg_surface_dataset], dtype=object)

np.save('../GoIS_dataset', gait_ireg_surface_dataset)

# code to load in np array
# gait_ireg_surface_dataset = np.load('gait_ireg_surface_dataset.npy', allow_pickle=True)

