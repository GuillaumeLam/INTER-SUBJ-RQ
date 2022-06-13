import numpy as np

with open('../data/X.npy', 'rb') as f:
	X = np.load(f)

print(X)
