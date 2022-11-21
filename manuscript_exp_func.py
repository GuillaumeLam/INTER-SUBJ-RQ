# Participant Calibration Curve (PaCalC)

import numpy as np
from sklearn.metrics import f1_score

import copy
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

labels = ['BnkL','BnkR', 'CS', 'FE', 'GR', 'SlpD', 'SlpU', 'StrD', 'StrU']

#======================>
#  Exported Functions  >
#======================>

# p_calib_curve: generate F1 vs C_tr curves per label type for single participant
# in:
#	-model
#	-participant features (X)
#	-participant labels	(Y)
# out:
#	-array of F1 vs C_tr per label type; dim:|unique(Y)| x max(|C_tr|)
def p_calib_curve(model, P_X, P_Y):
	return None

# p_calib_curve: generate F1 vs C_tr curves per label type for all participants
# in:
#	-model
#	-dataset features (X)
#	-dataset labels	(Y)
#	-dataset participant id (P)
# out:
#	-particpant-averaged array of F1 vs C_tr per label type; dim:|unique(Y)| x max(|C_tr|)
def avg_calib_curve(model,X,Y,P):
	# repeat p_calib_curve over all participant
	# average over participants (pad with last value [assumption: last value is highest] for shorter F1 vs C_tr arrays for all labels)
	return None

# graph_calib_curve_per_Y: generate detailed graph of F1 vs C_tr per label type
# in: 
#	-F1 vs C_tr curves; dim:|unique(Y)| x max(|C_tr|)
# out:
#	-graph of F1 vs C_tr per label type; dim: |unique(Y)|
def graph_calib_curve_per_Y(curves):
	return None

# graph_calib_curve_per_Y: generate graph of F1 vs C_tr averaged over label type
# in: 
#	-F1 vs C_tr curves; dim:|unique(Y)| x max(|C_tr|)
# out:
#	-graph of F1 vs C_tr; dim: 1
def graph_calib(curves):
	return None

#======================>
#  Internal Functions  >
#======================>

def per_label_calib(sw_tr_model, sw_te):
	X_te, Y_te, P_te = sw_te

	participants_dict = {}

	for i,_id_ in enumerate(P_te):
		if not _id_ in participants_dict:
			participants_dict[_id_] = ([],[]) # (P_X, P_Y)

		participants_dict[_id_][0].append(X_te[i])
		participants_dict[_id_][1].append(Y_te[i])

	for p_id in participants_dict.keys():
		per_label_dict, min_cycles = perLabelDict(participants_dict[p_id])

		print('P_id:',p_id)
		print('MIN cycles:',min_cycles)

		# for label in per_label_dict.keys():
			# train model on 1..n gait cycles & eval on else

	# order participants by min # per label

	# per n of calib gait cycles per label, train on n & eval on rest

	# 

# def repeated_eval(model, train_set, test_set, eval_schedule):

# def graph_re()

#====================>
#  Helper Functions  >
#====================>

# perLabelDict: make dict of gait cycles per label of participant
# return: 
# 	-dict of gait cycles per label, 
# 	-min number of gait cycles of all labels
def perLabelDict(P_XY):
	P_X, P_Y = P_XY
	label_dict = {}

	for i, y in enumerate(P_Y):
		print
		o_h_surface = y.argmax()
		if not o_h_surface in label_dict:
			label_dict[o_h_surface] = []

		label_dict[o_h_surface].append(P_X[i])

	min_cycles = 100000

	for i in range(0,len(labels)):
		if min_cycles > np.array(label_dict[i]).shape[0]:
			min_cycles = np.array(label_dict[i]).shape[0]

	return label_dict, min_cycles