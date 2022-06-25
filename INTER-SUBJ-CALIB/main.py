import os
import sys
import numpy as np

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from util.subject_wise_split import subject_wise_split

# ngois = np.load('../dataset/GoIS_dataset.npy', allow_pickle=True)
X = np.load('../dataset/GoIS_X_norm.npy', allow_pickle=True)
Y = np.load('../dataset/GoIS_Y.npy', allow_pickle=True)
P = np.load('../dataset/GoIS_P.npy', allow_pickle=True)

x_train, y_train, x_test, y_test, p_train, p_test = subject_wise_split(X,Y, participant=P, subject_wise=True,split=0.1,seed=42)
# print('Successful subject wise split!')
# print('p_train shape:')
# print(p_train.shape)
# print('p_test shape:')
# print(p_test.shape)

# print('x_train shape:')
# print(x_train.shape)
# print('x_test shape:')
# print(x_test.shape)

# print('y_train shape:')
# print(y_train.shape)
# print('y_test shape:')
# print(y_test.shape)

# print('p[n] val:')
# print(np.array(p_train))
# print(np.array(p_test))
# print('x[n] shape:')
# print(x_train[0].shape)
# print('y[n] val:')
# print(y_train[0].shape)

x_train, y_train, x_val, y_val, p_train, p_val = subject_wise_split(x_train,y_train, participant=p_train, subject_wise=True,split=0.1,seed=42)

print('Generated train, val, and test sets')

from models.regrML import REGR_model,REGR_compile

import tensorflow as tf

model = REGR_model('FFN_flat', x_train.shape[1:], y_train.shape[1:], verbose=True)
model = REGR_compile(model, 'categorical_crossentropy')

print('Model Generated')

model.fit(x_train,y_train, epochs=20,
			batch_size=2,
			validation_data=(x_val,y_val),
			callbacks=[
				tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min",restore_best_weights=True)
			])