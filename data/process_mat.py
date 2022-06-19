# script for py_compatible_data.mat => X.npy
import scipy.io
import numpy as np

data = scipy.io.loadmat('py_compatible_data.mat')
X = data['simplified_data']
X = np.array([[e[0][0][0], e[1], e[2][0][0][0]] for e in X], dtype=object)

np.save('X', X)

# code to load in np array
# X = np.load('X.npy', allow_pickle=True)

