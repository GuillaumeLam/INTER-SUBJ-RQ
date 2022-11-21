# Participant Calibration Curve (PaCalC)

import sys

import numpy as np
from sklearn.metrics import f1_score

import copy
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

#======================>
#  Exported Functions  >
#======================>

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# partiç_calib_curve: generate F1 vs C_tr curves per label type for single participant
# in:
#	-model
#	-participant features (X)
#	-participant labels	(Y)
# out:
#	-array of F1 vs C_tr per label type; dim:|unique(Y)| x max(|C_tr|)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def partic_calib_curve(model, P_X, P_Y):
	per_label_dict, min_cycles = perLabelDict(P_X, P_Y)

	f1_curves_per_label = []
	n_labels = len(per_label_dict.keys())

	i = 1

	for label in per_label_dict.keys():
		# order per_label_dict.keys()

		# train model on 1..n gait cycles & eval on else (always keep min 10% for eval)
			# X -> per_label_dict[label]
			# Y -> one_hot(label, n_labels) * len(X)

		f1_curve = [(8+i)]*min_cycles # TO REPLACE
		i+=1
		f1_curves_per_label.append(f1_curve)

	f1_matrix = pad_last_dim(f1_curves_per_label)

	return f1_matrix

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# partic_calib_curve: generate F1 vs C_tr curves per label type for all participants
# in:
#	-model
#	-dataset features (X)
#	-dataset one hot labels	(Y)
#	-dataset participant id (P)
# out:
#	-particpant-averaged array of F1 vs C_tr per label type; dim:|unique(Y)| x max(|C_tr|)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def avg_calib_curve(model,X,Y,P):
	participants_dict = perParticipantDict(X, Y, P)

	all_participants = []

	# repeat partic_calib_curve over all participant
	for p_id in participants_dict.keys():
		all_participants.append(partic_calib_curve(model,*participants_dict[p_id]))

	all_participants = pad_last_dim(all_participants)

	# average over participants (pad with last value [assumption: last value is highest] for shorter F1 vs C_tr arrays for all labels)
	avg_f1_matrix = np.mean(all_participants, axis=0)
	return avg_f1_matrix

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# graph_calib_curve_per_Y: generate detailed graph of F1 vs C_tr per label type
# in: 
#	-F1 vs C_tr curves; dim:|unique(Y)| x max(|C_tr|)
# out:
#	-graph of F1 vs C_tr per label type; dim: |unique(Y)|
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def graph_calib_curve_per_Y(curves, text_labels=None):
	return None

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# graph_calib_curve_per_Y: generate graph of F1 vs C_tr averaged over label type
# in: 
#	-F1 vs C_tr curves; dim:|unique(Y)| x max(|C_tr|)
# out:
#	-graph of F1 vs C_tr; dim: 1
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def graph_calib(curves, text_labels=None):
	return None

#====================>
#  Helper Functions  >
#====================>

# perLabelDict: make dict of gait cycles per label of participant
# in:
# 	-one hot labels
# return: 
# 	-dict of gait cycles per label, 
# 	-min number of gait cycles of all labels
def perLabelDict(P_X, P_Y):
	label_dict = {}

	for i, OHE_y in enumerate(P_Y):
		pos_y = OHE_y.argmax()
		if not pos_y in label_dict:
			label_dict[pos_y] = []

		label_dict[pos_y].append(P_X[i])

	for k in label_dict:
		P_X = label_dict[k]
		label_dict[k] = np.array(P_X)

	min_cycles = sys.maxsize

	for i in range(0,len(label_dict.keys())):
		if min_cycles > np.array(label_dict[i]).shape[0]:
			min_cycles = np.array(label_dict[i]).shape[0]

	return label_dict, min_cycles

# arr => array of array
def pad_last_dim(arr):
	# find longest length sub array
	l = 0
	for sub in arr:
		# print(np.array(sub).shape)
		l_sub = np.array(sub).shape[-1]
		if l_sub>l:
			l = l_sub

	sub_shape = np.array(arr[0]).shape

	# pad all sub arrays to longest sub array length with last subarray values
	matrix = np.empty((0 , *(() if len(sub_shape)==1 else np.array(arr[0]).shape[:-1]) , l))

	for sub in arr:
		sub = np.array(sub)
		l_sub = sub.shape[-1]

		if len(sub_shape)==1:
			padded_sub = np.append(sub, np.repeat(sub[...,-1],l-l_sub))
		else:
			padded_sub = np.hstack((sub, np.tile(sub[:,[-1]], l-l_sub)))
		matrix = np.append(matrix, np.array([padded_sub]), axis=0)

	return matrix

def perParticipantDict(X, Y, P):
	participants_dict = {}

	for i,_id_ in enumerate(P):
		if not _id_ in participants_dict:
			participants_dict[_id_] = ([],[]) # (P_X, P_Y)

		participants_dict[_id_][0].append(X[i])
		participants_dict[_id_][1].append(Y[i])

	for k in participants_dict:
		X,Y = participants_dict[k]
		participants_dict[k] = (np.array(X),np.array(Y))

	return participants_dict