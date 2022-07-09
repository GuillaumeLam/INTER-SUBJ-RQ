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

	return x_train, y_train, x_val, y_val, x_test, y_test, p_test

def gen_model(x_train, y_train, x_val, y_val):

	model = REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = REGR_model('Shah_FNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = REGR_model('CNN_L', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = AE_CNN(x_train, y_train)

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])

	print('Model Generated')

	# epochs = 20
	# batch_size = 2

	# epochs = 50
	# batch_size = 1

	epochs = 1
	batch_size = 512

	history = model.fit(x_train,y_train, epochs=epochs,
				batch_size=batch_size,
				validation_data=(x_val,y_val),
				callbacks=[
					tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min",restore_best_weights=True)
				]
				)

	# plt.clf()
	# plt.plot(history.history["loss"], label="Training Loss")
	# plt.plot(history.history["val_loss"], label="Validation Loss")
	# plt.legend()
	# plt.show()

	# plt.clf()
	# plt.plot(history.history["f1"], label="Training Loss")
	# plt.plot(history.history["val_f1"], label="Validation Loss")
	# plt.legend()
	# plt.show()

	return model

def gen_line(subject_wise):
	x_train, y_train, x_val, y_val, x_test, y_test, p_test = dataload(subject_wise=subject_wise)
	model = gen_model(x_train, y_train, x_val, y_val)

	mult_pred = model.predict(x_test)

	y_hat = np.zeros_like(mult_pred)
	y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

	labels = np.load('dataset/labels.npy', allow_pickle=True)

	print('subject_wise=',subject_wise)
	print(classification_report(y_test, y_hat,target_names=labels))

	report_dict = classification_report(y_test, y_hat,target_names=labels, output_dict=True)

	f1_line = []

	for l in labels:
		f1_line.append(report_dict[l]['f1-score'])

	return f1_line, (model, x_test, y_test, p_test, x_val, y_val)

def gen_graph():
	labels = np.load('dataset/labels.npy', allow_pickle=True)
	x = list(range(len(labels)))

	# add both to graph
	sw_f1, _ = gen_line(subject_wise=True)
	rw_f1, _ = gen_line(subject_wise=False)

	#display graph with model type
	plt.clf()
	plt.xticks(x,labels)
	plt.plot(x, sw_f1, label='subject-wise')	
	plt.plot(x, rw_f1, label='record-wise')	
	plt.legend()
	plt.show()

	return

# calibrate subject-wise trained model with n_cycles and eval performance
# func: generate calibration and test dataset from {X,Y,P}_test
def calibrate(model, x_test, y_test, p_test, x_val, y_val, n_cycle_per_label=10):
	# [OP] find patients with most number of gait cycles per label and calib on them

	# split datasets based on patient id and label
	p_y_dict = {}

	for p_id in list(set(p_test)):
		p_y_dict[p_id] = {}
		for l in labels:
			p_y_dict[p_id][l] = [np.empty((0,)+x_test.shape[1:]), np.empty((0,)+y_test.shape[1:]), np.empty((0,)+p_test.shape[1:])]

	for i in range(len(p_test)):
		p_y_dict[p_test[i]][oh2label(y_test[i])][0] = np.append(p_y_dict[p_test[i]][oh2label(y_test[i])][0], [x_test[i]], axis=0)
		p_y_dict[p_test[i]][oh2label(y_test[i])][1] = np.append(p_y_dict[p_test[i]][oh2label(y_test[i])][1], [y_test[i]], axis=0)
		p_y_dict[p_test[i]][oh2label(y_test[i])][2] = np.append(p_y_dict[p_test[i]][oh2label(y_test[i])][2], [p_test[i]], axis=0)

	dict_stats(p_y_dict)

	# split all arry in dict
	calib = [np.empty((0,)+x_test.shape[1:]), np.empty((0,)+y_test.shape[1:]), np.empty((0,)+p_test.shape[1:])]
	test = [np.empty((0,)+x_test.shape[1:]), np.empty((0,)+y_test.shape[1:]), np.empty((0,)+p_test.shape[1:])]

	for k in p_y_dict:
		for k2 in p_y_dict[k]:
			# add n_cycles to calib and rest to test (keep at least 10% for test))
			total_l = p_y_dict[k][k2][0].shape[0]
			if n_cycle_per_label > 0.9*total_l:
				# add 90% to calib and 10% to test
				n_cycle_per_label = int(0.9*total_l)

			for i in range(3):
				calib[i] = np.append(calib[i], p_y_dict[k][k2][i][:n_cycle_per_label], axis=0)
				test[i] = np.append(test[i], p_y_dict[k][k2][i][n_cycle_per_label:], axis=0)

	print("Calib shape")
	print("X_calib shape:", calib[0].shape)
	print("Y_calib shape:", calib[1].shape)
	print("P_calib shape:", calib[2].shape)

	print("Test shape")
	print("X_test shape:", test[0].shape)
	print("Y_test shape:", test[1].shape)
	print("P_test shape:", test[2].shape)

	return calib, test

	# epochs = 50
	# batch_size = 1

	# history = model.fit(x_calib,y_calib, epochs=epochs,
	# 	batch_size=batch_size,
	# 	validation_data=(x_val,y_val),
	# 	callbacks=[
	# 		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min",restore_best_weights=True)
	# 	]
	# )

def gen_calib_graph():
	labels = np.load('dataset/labels.npy', allow_pickle=True)
	x = list(range(len(labels)))

	# add both to graph
	sw_f1, sw_params = gen_line(subject_wise=True)
	rw_f1, _ = gen_line(subject_wise=True)

	n = 10

	for i in range(10):
		n_f1_line = calibrate(*sw_params, n_cycle_per_label=n)

		n += 10


def dict_stats(d):
	from collections import Counter

	for k in d:
		print('P_id: ', k)

		l = []
		n = []
		for k2 in d[k]:
			l.append(k2)
			n.append(d[k][k2][0].shape[0])
		print(l)
		print(n)

labels = np.load('dataset/labels.npy', allow_pickle=True)	
oh2label = lambda one_hot: labels[np.argmax(one_hot)]
# label2oh = lambda label:	

if __name__ == '__main__':
	# todo
	# -show necessary calib to hit rnd-wise split from subj-wise split
	# -lda
	# -mag

	# gen_graph()

	# fnc calibrate() debug()
	# =======================
	x_train, y_train, x_val, y_val, x_test, y_test, p_test = dataload(subject_wise=True)
	# model = gen_model(x_train, y_train, x_val, y_val)
	model = REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True)

	calibrate(model,x_test,y_test,p_test, x_val, y_val, n_cycle_per_label=4)