import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from util.subject_wise_split import subject_wise_split

import sys
import numpy as np

ngois = np.load('../dataset/nGoIS_dataset.npy', allow_pickle=True)

x_train, y_train, x_test, y_test, p_train, p_test = subject_wise_split(ngois[:,1],ngois[:,2], participant=ngois[:,0], subject_wise=True,split=0.1,seed=42)
print('Successful subject wise split!')
print('p_train shape:')
print(p_train.shape)
print('p_test shape:')
print(p_test.shape)

print('y_train shape:')
print(y_train.shape)
print('y_test shape:')
print(y_test.shape)

print('x_train shape:')
print(x_train.shape)
print('x_test shape:')
print(x_test.shape)