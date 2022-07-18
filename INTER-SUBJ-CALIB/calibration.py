import copy
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from random import seed, randint
from sklearn.metrics import classification_report

from main import dataload, keras_model_cpy, labels, oh2label

from main import epochs, batch_size

from split_diff import graph_split_diff

import os
import sys

sys.path.append(os.getcwd())

from models.regrML import REGR_model
from models.AE_CNN import AE_CNN


upper = 100	
x_axis = list(range(10,upper+10,10))

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
def get_calib_point(model, x_test, y_test, p_test, x_val, y_val, n_cycle_per_label=4):
	calib_set, test_set = calibrate_sets(x_test, y_test, p_test, n_cycle_per_label=n_cycle_per_label)

	(x_calib, y_calib, p_calib), (x_calib_test, y_calib_test, p_calib_test) = calib_set, test_set	# keep P_{calib,test} for easy verification of split leak 
																									# -> len(P_calib)*len(P_test)~O(N)^2; need to compare every elem of array with other array

	history = model.fit(x_calib,y_calib, epochs=epochs,
		batch_size=batch_size,
		validation_data=(x_val,y_val),
		callbacks=[
			tf.keras.callbacks.EarlyStopping(monitor="val_f1", patience=5, mode="max",restore_best_weights=True)
		]
	)

	plt.clf()
	plt.plot(history.history["loss"], label="Training")
	plt.plot(history.history["val_loss"], label="Validation")
	plt.legend()
	plt.savefig('calib_loss_'+'('+model_id+',e='+str(epochs)+',bs='+str(batch_size)+',cs='+str(n_cycle_per_label)+')')

	plt.clf()
	plt.plot(history.history["f1"], label="Training")
	plt.plot(history.history["val_f1"], label="Validation")
	plt.legend()
	plt.savefig('calib_f1_'+'('+model_id+',e='+str(epochs)+',bs='+str(batch_size)+',cs='+str(n_cycle_per_label)+')')

	mult_pred = model.predict(x_calib_test)

	y_calib_hat = np.zeros_like(mult_pred)
	y_calib_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

	report_dict = classification_report(y_calib_test, y_calib_hat,target_names=labels, output_dict=True)

	calib_f1 = []

	for l in labels:
		calib_f1.append(report_dict[l]['f1-score'])

	return calib_f1

def get_calib_line(model, model_id, seed=39):
	######################################
	sw_model, (sw_f1, rw_f1) = graph_split_diff(model, model_id, seed=seed)

	label_f1 = []
	for _ in range(len(labels)):
		label_f1.append([])

	x_train, y_train, x_val, y_val, x_test, y_test, p_test = dataload(subject_wise=True, seed=seed)

	for n in x_axis:
		model_cpy = keras_model_cpy(sw_model)
		params_cpy = model_cpy, x_test, y_test, p_test, x_val, y_val
		calib_f1_n = get_calib_point(model_cpy, x_test, y_test, p_test, x_val, y_val, n_cycle_per_label=n)

		for i, l_f1 in enumerate(calib_f1_n):
			label_f1[i].append(l_f1)

	######################################
	# repeat on diff seeds for std
	######################################
	# -> returns 9x10, repeat n times: nx9x10
	######################################

	return label_f1, sw_f1, rw_f1, model_cpy


def get_f1_calib(label_f1, sw_f1, rw_f1, detail, model_id):

	# given: 
	# - (r/s)w_f1 w/ shape: 9xn
	# - label_f1 w/ shape: 9xnx10

	# print('SHOWING SHAPES')
	# print('label_f1')
	# print(label_f1.shape)
	# print('(r/s)w_f1')
	# print(rw_f1.shape)
	# print(sw_f1.shape)

	plt.clf()
	if detail:
		sw_avg_f1 = np.mean(sw_f1, axis=1)
		rw_avg_f1 = np.mean(rw_f1, axis=1)
		sw_std_f1 = np.std(sw_f1, axis=1)
		rw_std_f1 = np.std(rw_f1, axis=1)
		
		for i, l in enumerate(label_f1):
			avg_calib_f1 = np.mean(l, axis=0)
			std_calib_f1 = np.std(l, axis=0)

			plt.subplot(3,3,i+1)

			# plt.plot(x_axis, l)
			plt.plot(x_axis, avg_calib_f1)
			plt.fill_between(
				x_axis,
				avg_calib_f1-std_calib_f1,
				avg_calib_f1+std_calib_f1, 
				alpha=0.4
			)

			plt.xlabel('Calibration size')
			plt.ylabel('F1')
			plt.title(labels[i])
			
			# plt.plot(0, sw_f1[i], 'ro',label='subject-wise')
			# plt.plot(upper, rw_f1[i], 'go', label='random-wise')
			plt.plot(0, sw_avg_f1[i], 'ro',label='subject-wise')
			plt.plot(upper, rw_avg_f1[i], 'go', label='random-wise')
			plt.errorbar(0, sw_avg_f1[i], yerr=sw_std_f1[i])
			plt.errorbar(upper, rw_avg_f1[i], yerr=rw_std_f1[i])

			plt.legend()
			plt.grid(linestyle='--', linewidth=0.5)
		plt.tight_layout()
		plt.suptitle('F1 vs calibration size for surface types')
			
	else:
		label_f1 = np.array(label_f1)
		avg_calib_f1 = np.mean(label_f1, axis=(0,1))
		std_calib_f1 = np.std(label_f1, axis=(0,1))

		sw_f1 = np.mean(sw_f1, axis=1)
		rw_f1 = np.mean(rw_f1, axis=1)

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

	title = 'f1_vs_calib_size_per_label_'+('detail_' if detail else '')
	plt.savefig('./out/'+title+'('+model_id+',e='+str(epochs)+',bs='+str(batch_size)+')')

# if detail=True: generate graphs of f1 for all labels, else: generate graph of overall f1 of labels
# MOD: model
def graph_f1_calib(model, model_id, cv=1):
	seed(39)

	seeds = [randint(0,1000) for _ in range(0,cv)]

	label_f1 = np.empty((0,9,10))
	sw_f1 = np.empty((0,9))
	rw_f1 = np.empty((0,9))

	for i,s in enumerate(seeds):
		l, sw, rw, model_cpy = get_calib_line(model, model_id, seed=s)

		label_f1 = np.append(label_f1, [l], axis=0)
		sw_f1 = np.append(sw_f1, [sw], axis=0)
		rw_f1 = np.append(rw_f1, [rw], axis=0)
		print('DONE 1 CV fold, progress: '+str(i*100/len(seeds))+'%')

	label_f1.transpose(1,0,2)
	sw_f1.transpose(1,0)
	rw_f1.transpose(1,0)

	# given: 
	# - (r/s)w_f1 w/ shape: 9xn
	# - label_f1 w/ shape: 9xnx10	

	get_f1_calib(copy.deepcopy(label_f1), copy.deepcopy(sw_f1), copy.deepcopy(rw_f1), detail=True, model_id=model_id)
	get_f1_calib(copy.deepcopy(label_f1), copy.deepcopy(sw_f1), copy.deepcopy(rw_f1), detail=False, model_id=model_id)

	return model_cpy # returned last calibrated model

def gen_f1_calib_graph():
	model_id = 'Shah_CNN'
	model = REGR_model(model_id, verbose=True)
	graph_f1_calib(model, model_id)

def gen_f1_calib_models_graph():

	models = []

	models.append('Shah_CNN+')
	models.append('Shah_CNN')
	models.append('Shah_FNN')
	models.append('CNN_L')
	# models.append('BiLinear')
	# models.append(AE_CNN(x_train, y_train))

	# generate graphs of difference in f1 from splits for various model types
	for model_id in models:
		graph_f1_calib(REGR_model(model_id, verbose=True), model_id, cv=10)

if __name__ == '__main__':
	# ================
	# calibration.py
	# Example use case
	# ================

	# 1. grab you favorite keras/tensorflow model
	model_id = 'Shah_CNN'
	# 2. load model in graphing function
	calib_model = graph_f1_calib(REGR_model(model_id, verbose=True), model_id)

	# (3.) sanity check: verify prediction of model
	# note: calib_model will be model which has been trained with the highest n_cycles_per_label
	graph_split_diff(calib_model, eval_only=True)
	# random_wise split and calibrated subject_wise split should be ~equivalent