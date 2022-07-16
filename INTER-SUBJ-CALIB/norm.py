import tensorflow as tf
import numpy as np

#+++++++++++++++++++++++++++++++
# For funcs 'gen_model', 'get_f1_line': ^
#+++++++++++++++++++++++++++++++

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, plot_confusion_matrix

#+++++++++++++++++++++++++++++++
# For funcs 'get_f1_line': ^
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
# For funcs 'gen_model', 'get_f1_line': ^
#+++++++++++++++++++++++++++++++

import copy

#+++++++++++++++++++++++++++++++
# For funcs 'gen_calib_graph': ^
#+++++++++++++++++++++++++++++++


labels = np.load('dataset/labels.npy', allow_pickle=True)	
oh2label = lambda one_hot: labels[np.argmax(one_hot)]
# label2oh = lambda label:	

epochs = 50
batch_size = 2

def dataload(subject_wise, seed=42):
	X = np.load('dataset/GoIS_X_norm.npy', allow_pickle=True)
	Y = np.load('dataset/GoIS_Y_norm.npy', allow_pickle=True)
	P = np.load('dataset/GoIS_P_norm.npy', allow_pickle=True)

	x_train, y_train, x_test, y_test, p_train, p_test = subject_wise_split(X, Y, participant=P, subject_wise=subject_wise,split=0.1,seed=seed)

	x_train, y_train, x_val, y_val, p_train, p_val = subject_wise_split(x_train, y_train, participant=p_train, subject_wise=True,split=0.1,seed=seed)

	print('Generated train, val, and test sets')

	# print('Size training set')
	# print(x_train.shape)
	# print('Size testing set')
	# print(x_test.shape)
	# print('Size overall set')
	# print(X.shape)

	return x_train, y_train, x_val, y_val, x_test, y_test, p_test

def gen_model(model, x_train, y_train, x_val, y_val):

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])

	print('Model Generated')

	# epochs = 20
	# batch_size = 2

	# epochs = 50
	# batch_size = 1

	# epochs = 1
	# batch_size = 512

	# print("CHECK: monitor val_loss or val_f1? (or else)")
	# history = model.fit(x_train,y_train, epochs=epochs,
	# 			batch_size=batch_size,
	# 			validation_data=(x_val,y_val),
	# 			callbacks=[
	# 				tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min",restore_best_weights=True)
	# 			]
	# 			)

	history = model.fit(x_train,y_train, epochs=epochs,
				batch_size=batch_size,
				validation_data=(x_val,y_val),
				callbacks=[
					tf.keras.callbacks.EarlyStopping(monitor="val_f1", patience=5, mode="min",restore_best_weights=True)
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

def keras_model_cpy(model):
	model_cpy = tf.keras.models.clone_model(model)
	model_cpy.build(model.input.shape)
	model_cpy.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
	model_cpy.set_weights(model.get_weights())

	return model_cpy

# +++++++++++++++++++++++++++++++++++++++
# 
# functions related to split difference graphs
#
# +++++++++++++++++++++++++++++++++++++++

# model can either be a keras/tensorflow model or a function f where f(X_tr,Y_tr,X_ts,Y_ts)-> model/f^theta
def get_f1_line(subject_wise, model_f):
	x_train, y_train, x_val, y_val, x_test, y_test, p_test = dataload(subject_wise=subject_wise)

	if callable(model_f):
		model = model_f(x_train, y_train, x_val, y_val)
	else:
		model = model_f

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

	return f1_line, model

# NON-MOD
def gen_split_diff_graph(model, show_wait=False):		
	x = list(range(len(labels)))

	f = lambda x_train, y_train, x_val, y_val: gen_model(keras_model_cpy(model), x_train, y_train, x_val, y_val)

	# add both to graph
	sw_f1, sw_model = get_f1_line(subject_wise=True, model_f=f)
	rw_f1, rw_model = get_f1_line(subject_wise=False, model_f=f)

	if not show_wait:
		plt.clf()
	plt.xticks(x,labels)
	# plt.plot(x, sw_f1, label='subject-wise')	
	# plt.plot(x, rw_f1, label='record-wise')

	plt.plot(sw_f1, label='subject-wise')	
	plt.plot(rw_f1, label='record-wise')
	if not show_wait:
		plt.legend()
		plt.show()

	return sw_model, (sw_f1, rw_f1)

def gen_shah_grah():
	model = REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	gen_split_diff_graph(model)

def gen_split_diff_graph_models():
	# model = REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = REGR_model('Shah_FNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = REGR_model('CNN_L', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = AE_CNN(x_train, y_train)

	models = []

	models.append(REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True))
	models.append(REGR_model('Shah_FNN', x_train.shape[1:], y_train.shape[1:], verbose=True))
	models.append(REGR_model('CNN_L', x_train.shape[1:], y_train.shape[1:], verbose=True))
	models.append(AE_CNN(x_train, y_train))

	# generate graphs of difference in f1 from splits for various model types
	for model in models:
		gen_split_diff_graph(model)

# +++++++++++++++++++++++++++++++++++++++
# 
# functions related to calibration graphs
#
# +++++++++++++++++++++++++++++++++++++++

def dict_stats(d):
	for k in d:
		print('P_id: ', k)

		l = []
		n = []
		for k2 in d[k]:
			l.append(k2)
			n.append(d[k][k2][0].shape[0])
		print(l)
		print(n)

# func: generate calibration and test dataset from {X,Y,P}_test
# NON-MOD
def calibrate_sets(x_te, y_te, p_te, n_cycle_per_label=10, debug=False):
	# [OP] find patients with most number of gait cycles per label and calib on them

	if debug:
		print('='*35)
		print('n=',n_cycle_per_label)
 
	x_test = copy.deepcopy(x_te)
	y_test = copy.deepcopy(y_te)
	p_test = copy.deepcopy(p_te)

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

	if debug:
		dict_stats(p_y_dict)

	# split all arry in dict
	calib = [np.empty((0,)+x_test.shape[1:]), np.empty((0,)+y_test.shape[1:]), np.empty((0,)+p_test.shape[1:])]
	test = [np.empty((0,)+x_test.shape[1:]), np.empty((0,)+y_test.shape[1:]), np.empty((0,)+p_test.shape[1:])]

	if debug:
		missing_samples_count = []

	for p_id in p_y_dict:
		for label in p_y_dict[p_id]:
			total_l = p_y_dict[p_id][label][0].shape[0]

			
			if n_cycle_per_label > 0.9*total_l:
				# keep at least 10% for testing
				# could change to only keep one sample for testing
				l = int(0.9*total_l)
				if debug:
					# print('[FLAG]: capping to leave 10% for p_id:'+str(p_id)+' and label:'+label)
					missing_samples_count.append(label)
			else:
				l = n_cycle_per_label

			for i in range(3):
				calib[i] = np.append(calib[i], p_y_dict[p_id][label][i][:l], axis=0)
				test[i] = np.append(test[i], p_y_dict[p_id][label][i][l:], axis=0)

	if debug:
		print('Under-represented labels with count:')
		v, c = np.unique(missing_samples_count, return_counts=True)
		print(v)
		print(c)
		print('Max Under-represented count: ',len(list(set(p_test))))

		print('CALIB abs size:', calib[2].shape[0])
		print('TEST abs size:', test[2].shape[0])

		print('CALIB set size: ' + str(calib[2].shape[0]*100/(calib[2].shape[0]+test[2].shape[0])) + '%')
		print('TEST set size: ' + str(test[2].shape[0]*100/(calib[2].shape[0]+test[2].shape[0])) + '%')

		print('ACTUAL to THEORECTIC CALIB size: '+str(calib[2].shape[0]*100/(n_cycle_per_label*calib[1].shape[1]*len(list(set(p_test)))))+'%')
		
		print('='*35)

	return calib, test

# MOD: model
def get_calib_line(model, x_test, y_test, p_test, x_val, y_val, n_cycle_per_label=4):
	calib_set, test_set = calibrate_sets(x_test, y_test, p_test, n_cycle_per_label=n_cycle_per_label)

	(x_calib, y_calib, p_calib), (x_calib_test, y_calib_test, p_calib_test) = calib_set, test_set	# keep P_{calib,test} for easy verification of split leak 
																									# -> len(P_calib)*len(P_test)~O(N)^2; need to compare every elem of array with other array

	# epochs = 50
	# batch_size = 1

	# print(x_calib.shape)
	# print(y_calib.shape)

	history = model.fit(x_calib,y_calib, epochs=epochs,
		batch_size=batch_size,
		validation_data=(x_val,y_val),
		callbacks=[
			tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min",restore_best_weights=True)
		]
	)

	# plt.clf()
	# plt.plot(history.history["f1"], label="Training Loss")
	# plt.plot(history.history["val_f1"], label="Validation Loss")
	# plt.legend()
	# plt.show()

	mult_pred = model.predict(x_calib_test)

	y_calib_hat = np.zeros_like(mult_pred)
	y_calib_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

	report_dict = classification_report(y_calib_test, y_calib_hat,target_names=labels, output_dict=True)

	calib_f1 = []

	for l in labels:
		calib_f1.append(report_dict[l]['f1-score'])

	return calib_f1

# if detail=True: generate graphs of f1 for all labels, else: generate graph of overall f1 of labels
# MOD: model
def gen_calib_graph(model, detail=True):
	######################################
	x_train, y_train, x_val, y_val, x_test, y_test, p_test = dataload(subject_wise=True, seed=39)
	# model = gen_model(model, x_train, y_train, x_val, y_val)

	labels = np.load('dataset/labels.npy', allow_pickle=True)

	# upper = 100
	upper = 50
	x_axis = list(range(10,upper+10,10))

	plt.clf()
	
	sw_model, (sw_f1, rw_f1) = gen_split_diff_graph(model, show_wait=True)
	# sw_model = model

	if detail:
		label_f1 = []
		for _ in range(len(labels)):
			label_f1.append([])
	else:
		avg_f1 = []
		std_f1 = []

	for n in x_axis:
		model_cpy = keras_model_cpy(sw_model)
		params_cpy = model_cpy, x_test, y_test, p_test, x_val, y_val
		calib_f1_n = get_calib_line(model_cpy, x_test, y_test, p_test, x_val, y_val, n_cycle_per_label=n)

		if detail:
			for i, l_f1 in enumerate(calib_f1_n):
				label_f1[i].append(l_f1)
		else:
			avg_f1.append(np.mean(np.array(calib_f1_n)))
			std_f1.append(np.std(np.array(calib_f1_n)))

	######################################
	# repeat on diff seeds for std
	######################################
	# -> returns 9x10, repeat n times: nx9x10
	######################################

	# given: 
	# - (r/s)w_f1 w/ shape: 9xn
	# - label_f1 w/ shape: 9xnx10

	plt.clf()
	if detail:
		# sw_avg_f1 = np.mean(sw_f1, axis=1)
		# rw_avg_f1 = np.mean(rw_f1, axis=1)
		# sw_std_f1 = np.std(sw_f1, axis=1)
		# rw_std_f1 = np.std(rw_f1, axis=1)
		
		for i, l in enumerate(label_f1):
			# avg_calib_f1 = np.mean(l, axis=0)
			# std_calib_f1 = np.std(l, axis=0)

			plt.subplot(3,3,i+1)

			plt.plot(x_axis, l)
			# plt.plot(x_axis, avg_calib_f1)
			# plt.fill_between(
			# 	x_axis,
			# 	avg_calib_f1-std_calib_f1,
			# 	avg_calib_f1+std_calib_f1, 
			# 	alpha=0.4
			# )

			plt.xlabel('Calibration size')
			plt.ylabel('F1')
			plt.title(labels[i])
			
			plt.plot(0, sw_f1[i], 'ro',label='subject-wise')
			plt.plot(upper, rw_f1[i], 'go', label='random-wise')
			# plt.plot(0, sw_avg_f1[i], 'ro',label='subject-wise')
			# plt.plot(upper, rw_avg_f1[i], 'go', label='random-wise')
			# plt.errorbar(0, sw_avg_f1[i], yerr=sw_std_f1[i])
			# plt.errorbar(upper, rw_avg_f1[i], yerr=rw_std_f1[i])

			plt.legend()
			plt.grid(linestyle='--', linewidth=0.5)
		plt.tight_layout()
		plt.suptitle('F1 vs calibration size for surface types')
			
	else:
		avg_calib_f1=np.array(avg_f1)
		std_calib_f1=np.array(std_f1)

		sw_f1 = np.array(sw_f1)
		rw_f1 = np.array(rw_f1)

		# label_f1 = np.array(label_f1)
		# avg_calib_f1 = np.mean(label_f1, axis=(0,1))
		# std_calib_f1 = np.std(label_f1, axis=(0,1))

		# sw_f1 = np.mean(sw_f1, axis=1)
		# rw_f1 = np.mean(rw_f1, axis=1)

		# display scaling of f1 => plt.plot(x_axis, avg_f1)
		plt.plot(x_axis, avg_calib_f1)
		plt.fill_between(
			x_axis,
			avg_calib_f1-std_calib_f1,
			avg_calib_f1+std_calib_f1, 
			alpha=0.4
		)

		plt.plot(0, np.mean(sw_f1), 'ro',label='subject-wise')
		plt.plot(upper, np.mean(rw_f1), 'go', label='random-wise')
		plt.errorbar(0, np.mean(sw_f1), yerr=np.std(sw_f1))
		plt.errorbar(upper, np.mean(rw_f1), yerr=np.std(rw_f1))

		plt.legend()
		plt.grid(linestyle='--', linewidth=0.5)
		plt.xlabel('Calibration size')
		plt.ylabel('F1')
		plt.title('F1 vs calibration size for surface types')
		# add title: f1 vs various calibration sizes

	plt.show()

	return model_cpy # returned last calibrated model

def gen_calib_shah_graph():
	model = REGR_model('Shah_CNN', verbose=True)
	gen_calib_graph(model)


if __name__ == '__main__':
	# todo
	# -show necessary calib to hit rnd-wise split from subj-wise split
	# -transfer data from training => calibrate ie. find min training size necessary
	# -mag
	# -lda

	# fnc calibrate_sets() debug()
	# =======================

	gen_calib_shah_graph()

	# ================
	# split_diff.py
	# Example use case
	# ================

	# # 1. grab you favorite keras/tensorflow model
	# model = REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# # 2. load model in graphing function
	# gen_split_diff_graph(model)


	# ================
	# calibration.py
	# Example use case
	# ================

	# # 1. grab you favorite keras/tensorflow model
	# model = REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# # 2. load model in graphing function
	# gen_calib_graph(model)

	# # (3.) sanity check: verify prediction of model
	# # note: model will be model which has been trained with the highest n_cycles_per_label
	# gen_split_diff_graph(model)
	# # random_wise split and calibrated subject_wise split should be equivalent