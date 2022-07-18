import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import wandb
from wandb.keras import WandbCallback

from main import dataload, gen_model, keras_model_cpy, labels, epochs, batch_size

import os
import sys

sys.path.append(os.getcwd())

from models.regrML import REGR_model
from models.AE_CNN import AE_CNN

# +++++++++++++++++++++++++++++++++++++++
# 
# functions related to split difference graphs
#
# +++++++++++++++++++++++++++++++++++++++

# model can either be a keras/tensorflow model or a function f where f(X_tr,Y_tr,X_ts,Y_ts)-> model/f^theta
def get_f1_line(subject_wise, model_f, seed=39):
	x_train, y_train, x_val, y_val, x_test, y_test, p_test = dataload(subject_wise=subject_wise, seed=seed)

	if callable(model_f):
		model = model_f(x_train, y_train, x_val, y_val)
	else:
		model = model_f

	mult_pred = model.predict(x_test)

	y_hat = np.zeros_like(mult_pred)
	y_hat[np.arange(len(mult_pred)), mult_pred.argmax(1)] = 1

	print('subject_wise=',subject_wise)
	print(classification_report(y_test, y_hat,target_names=labels))

	report_dict = classification_report(y_test, y_hat,target_names=labels, output_dict=True)

	f1_line = []

	for l in labels:
		f1_line.append(report_dict[l]['f1-score'])

	return f1_line, model

# NON-MOD
def graph_split_diff(model, model_id, eval_only=False, seed=39):		
	x = list(range(len(labels)))

	if not eval_only:
		f = lambda x_train, y_train, x_val, y_val: gen_model(keras_model_cpy(model), model_id, x_train, y_train, x_val, y_val)
	else:
		f = model

	# add both to graph
	wandb.init(project='inter-subject-calibration', entity='guillaumelam')
	wandb.run.name = 'init-train_subject-wise_'+model_id+'-e='+str(epochs)+'-bs='+str(batch_size)
	wandb.config = {
		'epochs': epochs,
		'batch_size': batch_size,
		'model_id': model_id
	}
	sw_f1, sw_model = get_f1_line(subject_wise=True, model_f=f, seed=seed)

	wandb.init(project='inter-subject-calibration', entity='guillaumelam')
	wandb.run.name = 'init-train_random-wise_'+model_id+'-e='+str(epochs)+'-bs='+str(batch_size)
	wandb.config = {
		'epochs': epochs,
		'batch_size': batch_size,
		'model_id': model_id
	}
	rw_f1, rw_model = get_f1_line(subject_wise=False, model_f=f, seed=seed)

	# if # models & f1 lines stored, load them

	# if not show_wait:
	plt.clf()
	plt.xticks(x,labels)

	plt.plot(sw_f1, label='subject-wise')	
	plt.plot(rw_f1, label='record-wise')
	# plt.fill_between(
	# 		x,
	# 		avg_calib_f1-std_calib_f1,
	# 		avg_calib_f1+std_calib_f1, 
	# 		alpha=0.4
	# 	)
	plt.legend()
	plt.savefig('./out/'+'sw_vs_rw_split_'+('eval_' if eval_only else '')+'('+model_id+',e='+str(epochs)+',bs='+str(batch_size)+')')
	# plt.show()

	return sw_model, (sw_f1, rw_f1)

def gen_shah_graph():
	model = REGR_model('Shah_CNN', verbose=True)
	graph_split_diff(model, 'Shah_CNN')

def gen_split_diff_models_graph():
	# model = REGR_model('Shah_CNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = REGR_model('Shah_FNN', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = REGR_model('CNN_L', x_train.shape[1:], y_train.shape[1:], verbose=True)
	# model = AE_CNN(x_train, y_train)

	models = []

	models.append('Shah_CNN')
	models.append('Shah_FNN')
	models.append('CNN_L')
	# models.append(AE_CNN(x_train, y_train))

	# generate graphs of difference in f1 from splits for various model types
	for model_id in models:
		graph_split_diff(REGR_model(model_id, verbose=True), model_id)


if __name__ == '__main__':
	# ================
	# split_diff.py
	# Example use case
	# ================

	# 1. grab you favorite keras/tensorflow model
	model_id = 'Shah_CNN'
	model = REGR_model(model_id, x_train.shape[1:], y_train.shape[1:], verbose=True)
	# 2. load model in graphing function
	graph_split_diff(model, model_id)