import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import wandb
from wandb.keras import WandbCallback

from subject_wise_split import subject_wise_split

import os
import sys

sys.path.append(os.getcwd())


from models.metrics import f1


labels = np.load('dataset/labels.npy', allow_pickle=True)	
oh2label = lambda one_hot: labels[np.argmax(one_hot)]

epochs = 50
batch_size = 4

# epochs = 2
# batch_size = 2048

def dataload(subject_wise, seed=42):
	X = np.load('dataset/GoIS_X_norm.npy', allow_pickle=True)
	Y = np.load('dataset/GoIS_Y_norm.npy', allow_pickle=True)
	P = np.load('dataset/GoIS_P_norm.npy', allow_pickle=True)

	x_train, y_train, x_test, y_test, p_train, p_test = subject_wise_split(X, Y, participant=P, subject_wise=subject_wise,split=0.1,seed=seed)

	# x_train, y_train, x_val, y_val, p_train, p_val = subject_wise_split(x_train, y_train, participant=p_train, subject_wise=True,split=0.1,seed=seed)

	print('Generated train, val, and test sets')

	# print('Size training set')
	# print(x_train.shape)
	# print('Size testing set')
	# print(x_test.shape)
	# print('Size overall set')
	# print(X.shape)

	return x_train, y_train, x_test, y_test, p_test

def gen_model(model, model_id, x_tr, y_tr, x_te, y_te):

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])

	print('Model Generated')

	config = wandb.config

	history = model.fit(x_tr,y_tr, epochs=config['epochs'],
				batch_size=config['batch_size'],
				validation_data=(x_te,y_te),
				callbacks=[
					tf.keras.callbacks.EarlyStopping(monitor="val_f1", patience=5, mode="max",restore_best_weights=True),
					WandbCallback()
				]
				)

	# plt.clf()
	# plt.plot(history.history["loss"], label="Training")
	# plt.plot(history.history["val_loss"], label="Validation")
	# plt.legend()
	# plt.savefig('./out/model_gen_loss_'+'('+model_id+',e='+str(epochs)+',bs='+str(batch_size)+')')

	# plt.clf()
	# plt.plot(history.history["f1"], label="Training")
	# plt.plot(history.history["val_f1"], label="Validation")
	# plt.legend()
	# plt.savefig('./out/model_gen_f1_'+'('+model_id+',e='+str(epochs)+',bs='+str(batch_size)+')')

	# plt.show()

	return model

def keras_model_cpy(model):
	model_cpy = tf.keras.models.clone_model(model)
	model_cpy.build(model.input.shape)
	model_cpy.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
	model_cpy.set_weights(model.get_weights())

	return model_cpy


if __name__ == '__main__':
	from split_diff import gen_shah_graph, gen_split_diff_models_graph
	from calibration import gen_f1_calib_graph, gen_f1_calib_models_graph

	# MINIMAL CODE TO GENERATE SHAH GRAPH
	# gen_shah_graph()

	# # MINIMAL CODE TO GENERATE SPLIT DIFFERENCE GRAPH FOR ALL MODELS 
	gen_split_diff_models_graph()

	# MINIMAL CODE TO GENERATE CALIBRATION GRAPH
	# gen_f1_calib_graph()

	# # MINIMAL CODE TO GENERATE SPLIT DIFFERENCE GRAPH FOR ALL MODELS 
	# gen_f1_calib_models_graph()