from load_data import load_surface_data, _CACHED_load_surface_data
from subject_wise_split import subject_wise_split
from cf import CF_DNN

import numpy as np
from sklearn.metrics import f1_score

import copy
from random import seed, randint
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

labels = ['BnkL','BnkR', 'CS', 'FE', 'GR', 'SlpD', 'SlpU', 'StrD', 'StrU']

def single_run(X_train, y_train, X_test, y_test):
	result = {}

	# hid_layers=(606,303,606) #hidden layers
	# model='classification' #problem type
	# output= y_train.shape[-1] #ouput shape 
	# input_shape=X_train.shape[-1]
	# ann=Lab.ANN(hid_layers=hid_layers,model=model,output=output,input_shape=input_shape,activation_hid='relu') # relu in hidden layers
	ann = CF_DNN(x_shape=(480), y_shape=9, model_shape=(606,303,606)).ann

	ann.fit(X_train,y_train,batch_size=512,epochs=50, validation_split=0.1)
	y_pred=ann.predict(X_test)
	a=np.zeros_like(y_pred)
	a[np.arange(len(y_pred)), y_pred.argmax(1)]=1
	y_pred=a
	b=ann.predict(X_train)
	a=b.argmax(axis=1)
	result['ACC_TR']=np.mean(y_train.argmax(axis=1)==a)

	result['F1']=f1_score(y_test, y_pred,average='weighted')
	result['F1_classes']=f1_score(y_test, y_pred,average=None)
	
	return result, ann, (X_test, y_test)

def keras_model_cpy(model):
	model_cpy = tf.keras.models.clone_model(model)
	model_cpy.build(model.input.shape)
	model_cpy.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model_cpy.set_weights(model.get_weights())

	return model_cpy

def get_calib_point(model, x_test, y_test, p_test, seed, calib_split=None):
	# if no calib_split specified, method essentially evals model and reports f1 per label

	if calib_split is not None:
		x_ctr, y_ctr, x_test, y_test, p_ctr, p_test = subject_wise_split(np.copy(x_test), np.copy(y_test), np.copy(p_test), split=0.1, seed=seed, subject_wise=False)
		x_calib, y_calib, _, _, _, _ = subject_wise_split(x_ctr, y_ctr, p_ctr, split=calib_split, seed=seed, subject_wise=False)
		if x_calib.shape[0] != 0:
			history = model.fit(x_calib, y_calib, epochs=50,batch_size=16,validation_split=0.1)

	mult_pred = model.predict(x_test)

	y_hat = np.zeros_like(mult_pred)
	y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

	report_dict = classification_report(y_test, y_hat,target_names=labels, output_dict=True)

	calib_f1 = []

	for l in labels:
		calib_f1.append(report_dict[l]['f1-score'])

	if calib_split is not None:
		return calib_f1, x_calib.shape[0], model
	else:
		return calib_f1, x_test.shape[0], model

# x_axis = [e*0.1 for e in list(range(0,10,1))]
x_axis = [0.0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

def calibrated_line(model, X_test, Y_test, P_test, seed, freeze):
	label_f1 = []
	for _ in range(len(labels)):
		label_f1.append([])

	calib_sizes = []
	for n in x_axis:
		print('Calibrating and evaluating n='+str(n))
		model_cpy = keras_model_cpy(model)

		if freeze:
			# freeze all layers but last layer of model
			print(model_cpy.layers)

			print(model_cpy.summary())
			for i in range(len(model_cpy.layers)):
				if i < 2:
					model_cpy.layers[i].trainable = False
			print(model_cpy.summary())

			# model_cpy.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		calib_f1_n, calib_size, calibrated_model = get_calib_point(model_cpy, X_test, Y_test, P_test, seed, 1-n)
		for i, l_f1 in enumerate(calib_f1_n):
			label_f1[i].append(l_f1)
		calib_sizes.append(calib_size)

	return label_f1, calib_sizes, calibrated_model

def calib_data(cv=1, freeze=False, holdout=False):
	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	seed(39)
	seeds = [randint(0,1000) for _ in range(0,cv)]

	label_f1 = np.empty((0,9,len(x_axis)))
	sw_f1 = np.empty((0,9))
	rw_f1 = np.empty((0,9))

	if holdout:
		holdout_f1 = np.empty((0,9))

	for i,s in enumerate(seeds):

		X_tr, Y_tr, P_tr, X_te, Y_te, P_te, _ = load_surface_data(s, False, split=0.1)
		rw_result, _, _ = single_run(X_tr, Y_tr, X_te, Y_te)

		X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(s, True, split=0.1)
		
		if holdout:
			X_tr, Y_tr, X_hold, Y_hold, P_tr, P_hold = subject_wise_split(X_tr, Y_tr, P_tr, split=0.1, seed=s, subject_wise=True)
		sw_result, sw_model, (X_te, Y_te) = single_run(lab, X_tr, Y_tr, X_te, Y_te)

		rw = rw_result['F1_classes']
		sw = sw_result['F1_classes']

		l, calib_sizes, calibrated_model = calibrated_line(sw_model, X_te, Y_te, P_te, s, freeze=freeze) # calibrated_model is model with highest calibration

		label_f1 = np.append(label_f1, np.array([l]), axis=0)
		sw_f1 = np.append(sw_f1, np.array([sw]), axis=0)
		rw_f1 = np.append(rw_f1, np.array([rw]), axis=0)

		# current dip with < ~100 gait cycles would seem to indicate that the model is greedily learning 
		# specifically gait pattern on indiv => test acc of model on third hold-out set should be as bad 
		# even after calib, indicating that calibration in no way solves prob of focusing on Z ie. patient
		# rather than Y ie. surface
		if holdout:
			print('+'*30)
			print("Evaluating calibrated model on holdout set")
			# eval calibrated model on third holdout set; expectation: very bad as calibration does not solve issue
			holdout_f1_n, _, _ = get_calib_point(calibrated_model, X_hold, Y_hold, P_hold, s)
			print(holdout_f1_n)
			print('+'*30)
			holdout_f1 = np.append(holdout_f1, np.array([holdout_f1_n]), axis=0)

		print('DONE 1 CV fold, progress: '+str((i+1)*100/len(seeds))+'%')

	# given: 
	# - (r/s)w_f1 w/ shape: 9 x f
	# - label_f1 w/ shape: 9 x f x s	

	if not holdout:
		return label_f1.transpose(1,0,2), sw_f1.transpose(1,0), rw_f1.transpose(1,0), np.array(calib_sizes)
	else:
		return label_f1.transpose(1,0,2), sw_f1.transpose(1,0), rw_f1.transpose(1,0), holdout_f1.transpose(1,0), np.array(calib_sizes)


def graph_f1_calib(label_f1, sw_f1, rw_f1, calib_sizes, detail, model_id, log_scale=False, freeze=False, holdout=False, std=True):

	# given: 
	# - (r/s)w_f1 w/ shape: 9 x f
	# - label_f1 w/ shape: 9 x f x s
	# - calib_sizes w/ shape: s

	cv=sw_f1.shape[1]

	plt.clf()
	if detail:
		sw_avg_f1 = np.mean(sw_f1, axis=1)
		rw_avg_f1 = np.mean(rw_f1, axis=1)
		sw_std_f1 = np.std(sw_f1, axis=1)
		rw_std_f1 = np.std(rw_f1, axis=1)

		for i, l in enumerate(label_f1):
			avg_calib_f1 = np.mean(l, axis=0)

			if std:
				std_calib_f1 = np.std(l, axis=0)
			else:
				min_calib_f1 = np.min(l, axis=0)
				max_calib_f1 = np.max(l, axis=0)

			plt.subplot(3,3,i+1)

			# plt.plot(x_axis, l)
			plt.plot(calib_sizes, avg_calib_f1)
			
			if std:
				plt.fill_between(
					calib_sizes,
					avg_calib_f1-std_calib_f1,
					avg_calib_f1+std_calib_f1, 
					alpha=0.4
				)
			else:
				plt.fill_between(
					calib_sizes,
					min_calib_f1,
					max_calib_f1, 
					alpha=0.4
				)

			plt.title(labels[i])
			
			if i == 3:
				plt.ylabel('F1')
			elif i == 7:
				plt.xlabel('Calibration size')

			# plt.plot(0, sw_f1[i], 'ro',label='subject-wise')
			# plt.plot(upper, rw_f1[i], 'go', label='random-wise')
			if i == 0:
				plt.plot(0, sw_avg_f1[i], 'go',label='subject-wise')
				plt.plot(calib_sizes[-1], rw_avg_f1[i], 'ro', label='random-wise')
			else:
				plt.plot(0, sw_avg_f1[i], 'go')
				plt.plot(calib_sizes[-1], rw_avg_f1[i], 'ro')
			
			plt.errorbar(0, sw_avg_f1[i], yerr=sw_std_f1[i], ecolor='green')
			plt.errorbar(calib_sizes[-1], rw_avg_f1[i], yerr=rw_std_f1[i], ecolor='red')

			plt.grid(linestyle='--', linewidth=0.5)
			if log_scale:
				plt.xscale('symlog')

		plt.figlegend()
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		plt.suptitle('F1 vs calibration size'+('(log)' if log_scale else '')+' per surface types')

	else:
		label_f1 = np.array(label_f1)
		avg_calib_f1 = np.mean(label_f1, axis=(0,1))

		if std:
			std_calib_f1 = np.std(label_f1, axis=(0,1))
		else:
			min_calib_f1 = np.min(label_f1, axis=(0,1))
			max_calib_f1 = np.max(label_f1, axis=(0,1))

		sw_f1 = np.mean(sw_f1, axis=1)
		rw_f1 = np.mean(rw_f1, axis=1)

		plt.plot(calib_sizes, avg_calib_f1)

		if std:
			plt.fill_between(
				calib_sizes,
				avg_calib_f1-std_calib_f1,
				avg_calib_f1+std_calib_f1,
				alpha=0.4
			)
		else:
			plt.fill_between(
				calib_sizes,
				min_calib_f1,
				max_calib_f1, 
				alpha=0.4
			)

		plt.plot(0, np.mean(sw_f1), 'go',label='subject-wise')
		plt.plot(calib_sizes[-1], np.mean(rw_f1), 'ro', label='random-wise')
		plt.errorbar(0, np.mean(sw_f1), ecolor='green', yerr=np.std(sw_f1))
		plt.errorbar(calib_sizes[-1], np.mean(rw_f1), ecolor='red', yerr=np.std(rw_f1))

		plt.legend(loc='lower right')
		plt.grid(linestyle='--', linewidth=0.5)
		plt.xlabel('Calibration size')
		if log_scale:
			plt.xscale('symlog')
		plt.ylabel('F1')
		plt.title('F1 vs calibration size'+('(log)' if log_scale else ''))

	title = 'f1_vs_calib_size_per_label_'+('detail_' if detail else '')
	epochs=50
	batch_size=16

	options = ''

	if freeze:
		options += ',freeze=True'

	if holdout:
		options += ',holdout=True'

	plt.savefig('./out/'+title+'('+model_id+',e='+str(epochs)+',bs='+str(batch_size)+',cv='+str(cv)+',log='+str(log_scale)+options+')')


def graph_split_diff(rw_f1, sw_f1, std):
	x = list(range(len(labels)))

	avg_sw_f1 = np.mean(sw_f1, axis=1)

	if std:
		std_sw_f1 = np.std(sw_f1, axis=1)
	else:
		min_sw_f1 = np.min(sw_f1, axis=1)
		max_sw_f1 = np.max(sw_f1, axis=1)
	
	avg_rw_f1 = np.mean(rw_f1, axis=1)

	if std:
		std_rw_f1 = np.std(rw_f1, axis=1)
	else:
		min_rw_f1 = np.min(rw_f1, axis=1)
		max_rw_f1 = np.max(rw_f1, axis=1)

	# if not show_wait:
	plt.clf()
	plt.xticks(x,labels)

	plt.plot(avg_sw_f1, label='subject-wise')	
	plt.plot(avg_rw_f1, label='record-wise')

	if std:
		plt.fill_between(
			x,
			avg_sw_f1-std_sw_f1,
			avg_sw_f1+std_sw_f1, 
			alpha=0.4
		)
		plt.fill_between(
			x,
			avg_rw_f1-std_rw_f1,
			avg_rw_f1+std_rw_f1,
			alpha=0.4
		)
	else:
		plt.fill_between(
			x,
			min_sw_f1,
			max_sw_f1, 
			alpha=0.4
		)
		plt.fill_between(
			x,
			min_rw_f1,
			max_rw_f1, 
			alpha=0.4
		)

def graph_holdout(holdout_f1, rw_f1, sw_f1, freeze=False, std=True):
	graph_split_diff(rw_f1, sw_f1, std)

	avg_holdout_f1 = np.mean(holdout_f1, axis=1)

	if std:
		std_holdout_f1 = np.std(holdout_f1, axis=1)
	else:
		min_holdout_f1 = np.min(holdout_f1, axis=1)
		max_holdout_f1 = np.max(holdout_f1, axis=1)

	x = list(range(len(labels)))

	if freeze:
		plt.plot(avg_holdout_f1, label='calibrated frozen model on subject-wise')
	else:
		plt.plot(avg_holdout_f1, label='calibrated model on subject-wise')

	if std:
		plt.fill_between(
			x,
			avg_holdout_f1-std_holdout_f1,
			avg_holdout_f1+std_holdout_f1,
			alpha=0.4
		)
	else:
		plt.fill_between(
			x,
			min_holdout_f1,
			max_holdout_f1, 
			alpha=0.4
		)

	plt.legend()
	plt.title('F1 vs surface types')

	plt.savefig('./out/'+'sw_vs_rw_split_with_holdout'+('(freeze=True)' if freeze else ''))