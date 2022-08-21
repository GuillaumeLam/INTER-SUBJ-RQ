from load_data import load_surface_data
from subject_wise_split import subject_wise_split

import numpy as np
from sklearn.metrics import f1_score

import copy
from random import seed, randint
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

labels = ['BnkL','BnkR', 'CS', 'FE', 'GR', 'SlpD', 'SlpU', 'StrD', 'StrU']

def single_run(Lab, X_train, y_train, X_test, y_test):
	result = {}

	hid_layers=(606,303,606) #hidden layers
	model='classification' #problem type
	output= y_train.shape[-1] #ouput shape 
	input_shape=X_train.shape[-1]
	ann=Lab.ANN(hid_layers=hid_layers,model=model,output=output,input_shape=input_shape,activation_hid='relu') # relu in hidden layers
	
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

def get_calib_point(model, X_test, Y_test, P_test, seed, calib_split):
	x_ctr, y_ctr, x_test, y_test, p_ctr, p_test = subject_wise_split(X_test, Y_test, P_test, split=0.1, seed=seed, subject_wise=False)
	x_calib, y_calib, _, _, _, _ = subject_wise_split(x_ctr, y_ctr, p_ctr, split=calib_split, seed=seed, subject_wise=False)
	print(x_calib.shape)
	if x_calib.shape[0] != 0:
		history = model.fit(x_calib, y_calib, epochs=50,batch_size=16,validation_split=0.1)

	mult_pred = model.predict(x_test)

	y_hat = np.zeros_like(mult_pred)
	y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

	report_dict = classification_report(y_test, y_hat,target_names=labels, output_dict=True)

	calib_f1 = []

	for l in labels:
		calib_f1.append(report_dict[l]['f1-score'])

	return calib_f1, x_calib.shape[0]

# x_axis = [e*0.1 for e in list(range(0,10,1))]
x_axis = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.5]

def calibrated_line(model, X_test, Y_test, P_test, seed):
	label_f1 = []
	for _ in range(len(labels)):
		label_f1.append([])

	calib_sizes = []
	for n in x_axis:
		print('Calibrating and evaluting n='+str(n))
		model_cpy = keras_model_cpy(model)
		calib_f1_n, calib_size = get_calib_point(model_cpy, X_test, Y_test, P_test, seed, 1-n)
		for i, l_f1 in enumerate(calib_f1_n):
			label_f1[i].append(l_f1)
		calib_sizes.append(calib_size)

	return label_f1, calib_sizes

def calib_data(cv=1):
	seed(39)
	seeds = [randint(0,1000) for _ in range(0,cv)]

	label_f1 = np.empty((0,9,len(x_axis)))
	sw_f1 = np.empty((0,9))
	rw_f1 = np.empty((0,9))

	for i,s in enumerate(seeds):

		X_tr, Y_tr, P_tr, X_te, Y_te, P_te, lab = load_surface_data(s, False)
		rw_result, _, _ = single_run(lab, X_tr, Y_tr, X_te, Y_te)

		X_tr, Y_tr, P_tr, X_te, Y_te, P_te, lab = load_surface_data(s, True)
		sw_result, sw_model, (X_te, Y_te) = single_run(lab, X_tr, Y_tr, X_te, Y_te)

		# rw = rw_result[s]['subject_wise_False']['F1score_classes']['Lower']
		# sw = sw_result[s]['subject_wise_True']['F1score_classes']['Lower']

		rw = rw_result['F1_classes']
		sw = sw_result['F1_classes']

		# labels= lab.surface_name

		# print("LABELS")
		# print(labels)

		l, calib_sizes = calibrated_line(sw_model, X_te, Y_te, P_te, s)

		label_f1 = np.append(label_f1, np.array([l]), axis=0)
		sw_f1 = np.append(sw_f1, np.array([sw]), axis=0)
		rw_f1 = np.append(rw_f1, np.array([rw]), axis=0)
		print('DONE 1 CV fold, progress: '+str((i+1)*100/len(seeds))+'%')

	# given: 
	# - (r/s)w_f1 w/ shape: 9xn
	# - label_f1 w/ shape: 9xnx10	

	return label_f1.transpose(1,0,2), sw_f1.transpose(1,0), rw_f1.transpose(1,0), np.array(calib_sizes)

def graph_f1_calib(label_f1, sw_f1, rw_f1, calib_sizes, detail, model_id, log_scale=False):

	# given: 
	# - (r/s)w_f1 w/ shape: 9xf
	# - label_f1 w/ shape: 9xfxs
	# - calib_sizes w/ shape: s

	# print('SHOWING SHAPES')
	# print('label_f1')
	# print(label_f1.shape)
	# print('(r/s)w_f1')
	# print(rw_f1.shape)
	# print(sw_f1.shape)

	# print(calib_sizes)
	# calib_sizes=np.array([e[0] for e in calib_sizes])

	cv=sw_f1.shape[1]

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
			plt.plot(calib_sizes, avg_calib_f1)
			plt.fill_between(
				calib_sizes,
				avg_calib_f1-std_calib_f1,
				avg_calib_f1+std_calib_f1, 
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
		plt.suptitle('F1 vs calibration size for surface types')

	else:
		label_f1 = np.array(label_f1)
		avg_calib_f1 = np.mean(label_f1, axis=(0,1))
		std_calib_f1 = np.std(label_f1, axis=(0,1))

		sw_f1 = np.mean(sw_f1, axis=1)
		rw_f1 = np.mean(rw_f1, axis=1)

		plt.plot(calib_sizes, avg_calib_f1)
		plt.fill_between(
			calib_sizes,
			avg_calib_f1-std_calib_f1,
			avg_calib_f1+std_calib_f1, 
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
		plt.title('F1 vs calibration size for surface types')

	title = 'f1_vs_calib_size_per_label_'+('detail_' if detail else '')
	epochs=50
	batch_size=16
	plt.savefig('./out/'+title+'('+model_id+',e='+str(epochs)+',bs='+str(batch_size)+',cv='+str(cv)+',log='+str(log_scale)+')')
