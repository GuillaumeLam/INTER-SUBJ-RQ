# script for py_compatible_data.mat => X.npy
import scipy.io
import numpy as np

data = scipy.io.loadmat('py_compatible_data.mat')
X = data['data']
X = np.array([[e[0][0][0], e[1], e[2][0]] for e in X])

with open('X.npy', 'wb') as f:
	np.save(f, X)

# code to load in np array
# with open('X.npy', 'rb') as f:
#	X = np.load(f)

