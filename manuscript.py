from load_data import load_surface_data, _CACHED_load_surface_data
from subject_wise_split import subject_wise_split

import numpy as np
from sklearn.metrics import f1_score

import copy
from random import seed, randint
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

labels = ['BnkL','BnkR', 'CS', 'FE', 'GR', 'SlpD', 'SlpU', 'StrD', 'StrU']

#======================
#  Exported Functions
#======================

def p_calib_curve(model, P_X, P_Y):
	return None

def avg_calib_curve(model,X,Y,P):
	return None

def graph_calib_curve_per_Y(curves):
	return None

def graph_calib(curves):
	return None

#====================
#  Helper Functions
#====================


# perLabel: make dict of gait cycles per label of participant
# return: 
# 		dict of gait cycles per label, 
# 		min number of gait cycles of all labels
def perLabel(P_XY):
	P_X, P_Y = P_XY
	label_dict = {}

	for i, y in enumerate(P_Y):
		print
		o_h_surface = y.argmax()
		if not o_h_surface in label_dict:
			label_dict[o_h_surface] = []

		label_dict[o_h_surface].append(P_X[i])

	min_cycles = 1000

	for i in range(0,len(labels)):
		if min_cycles > np.array(label_dict[i]).shape[0]:
			min_cycles = np.array(label_dict[i]).shape[0]

	return label_dict, min_cycles

def per_label_calib(sw_tr_model, sw_te):
	X_te, Y_te, P_te = sw_te

	participants_dict = {}

	for i,_id_ in enumerate(P_te):
		if not _id_ in participants_dict:
			participants_dict[_id_] = ([],[]) # (P_X, P_Y)

		participants_dict[_id_][0].append(X_te[i])
		participants_dict[_id_][1].append(Y_te[i])

	# perLabel(participants_dict[1])

	for p_id in participants_dict.keys():
		per_label_dict, min_cycles = perLabel(participants_dict[p_id])

		print('P_id:',p_id)
		print('MIN cycles:',min_cycles)

		# for label in per_label_dict.keys():
			# train model on 1..n gait cycles & eval on else

	# order participants by min # per label

	# per n of calib gait cycles per label, train on n & eval on rest

	# 

# def repeated_eval(model, train_set, test_set, eval_schedule):


# def graph_re()


if __name__ == "__main__":
	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(214, True, split=0.1)

	per_label_calib(None, (X_te, Y_te, P_te))