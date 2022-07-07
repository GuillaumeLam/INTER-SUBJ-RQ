import tensorflow as tf
import numpy as np

#+++++++++++++++++++++++++++++++
# For funcs 'gen_model', 'gen_line': ^
#+++++++++++++++++++++++++++++++

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, plot_confusion_matrix

#+++++++++++++++++++++++++++++++
# For funcs 'gen_line': ^
#+++++++++++++++++++++++++++++++

import os
import sys

sys.path.append(os.getcwd())

#+++++++++++++++++++++++++++++++
# To import proj functions: ^
#+++++++++++++++++++++++++++++++

from util.subject_wise_split import subject_wise_split

#+++++++++++++++++++++++++++++++
# For funcs 'dataload': ^
#+++++++++++++++++++++++++++++++

from models.regrML import REGR_model, REGR_compile
from models.metrics import f1
from models.AE_CNN import AE_CNN

#+++++++++++++++++++++++++++++++
# For funcs 'gen_model', 'gen_line': ^
#+++++++++++++++++++++++++++++++

def dataload(subject_wise):
	X = np.load('dataset/GoIS_X_norm.npy', allow_pickle=True)
	Y = np.load('dataset/GoIS_Y_norm.npy', allow_pickle=True)
	P = np.load('dataset/GoIS_P_norm.npy', allow_pickle=True)

	x_train, y_train, x_test, y_test, p_train, p_test = subject_wise_split(X,Y, participant=P, subject_wise=subject_wise,split=0.1,seed=42)

	x_train, y_train, x_val, y_val, p_train, p_val = subject_wise_split(x_train,y_train, participant=p_train, subject_wise=True,split=0.1,seed=42)

	print('Generated train, val, and test sets')

	# print('Size training set')
	# print(x_train.shape)
	# print('Size testing set')
	# print(x_test.shape)
	# print('Size overall set')
	# print(X.shape)

	return x_train, y_train, x_val, y_val, x_test, y_test

def gen_model(x_train, y_train, x_val, y_val):

	model = REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = REGR_model('Shah_FNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = REGR_model('CNN_L', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = AE_CNN(x_train, y_train)

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])

	print('Model Generated')

	# epochs = 20
	# batch_size = 2

	epochs = 50
	batch_size = 1

	# epochs = 3
	# batch_size = 64

	history = model.fit(x_train,y_train, epochs=epochs,
				batch_size=batch_size,
				validation_data=(x_val,y_val),
				callbacks=[
					tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min",restore_best_weights=True)
				]
				)

	plt.clf()
	plt.plot(history.history["loss"], label="Training Loss")
	plt.plot(history.history["val_loss"], label="Validation Loss")
	plt.legend()
	plt.show()

	plt.clf()
	plt.plot(history.history["f1"], label="Training Loss")
	plt.plot(history.history["val_f1"], label="Validation Loss")
	plt.legend()
	plt.show()

	return model

def gen_line(subject_wise):
	x_train, y_train, x_val, y_val, x_test, y_test = dataload(subject_wise=subject_wise)
	model = gen_model(x_train, y_train, x_val, y_val)

	mult_pred = model.predict(x_test)

	y_hat = np.zeros_like(mult_pred)
	y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

	labels = np.load('dataset/labels.npy', allow_pickle=True)

	print('subject_wise=',subject_wise)
	print(classification_report(y_test, y_hat,target_names=labels))

	report_dict = classification_report(y_test, y_hat,target_names=labels)

	f1_line = []

	for l in labels:
		f1_line.append(report_dict[l]['f1-score'])

	return f1_line

def gen_graph():
	labels = np.load('dataset/labels.npy', allow_pickle=True)
	x = list(range(len(labels)))

	# add both to graph
	sw_f1 = gen_line(subject_wise=True)
	rw_f1 = gen_line(subject_wise=False)

	#display graph with model type
	plt.clf()
	plt.xticks(x,labels)
	plt.plot(x, sw_f1, label='subject-wise')	
	plt.plot(x, rw_f1, label='record-wise')	
	plt.show()

	return


if __name__ == '__main__':
	# todo
	# -show necessary calib to hit rnd-wise split from subj-wise split
	# -lda
	# -mag

	gen_graph()
