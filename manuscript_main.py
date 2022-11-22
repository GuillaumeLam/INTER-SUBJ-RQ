from load_data import load_surface_data, _CACHED_load_surface_data
from subject_wise_split import subject_wise_split
import GaitLab2Go as GL2G

import numpy as np
from sklearn.metrics import f1_score

import copy
import pickle
from random import seed, randint
from sklearn.metrics import classification_report
import tensorflow as tf

import manuscript_exp_func as PaCalC

irreg_surfaces_labels = ['BnkL','BnkR', 'CS', 'FE', 'GR', 'SlpD', 'SlpU', 'StrD', 'StrU']

# TODO: 
#	-look into f1 of 1.0 very early, even for test data...? => rounding errors?, data leaking?, WRONG LABEL (no trials => 100%)
# 		-silence PaCalC keras calib warnings

def PaCalC_F1(dtst_seed=214, save=False):
	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1)

	ann = make_model(X_tr, Y_tr)

	# train model on X_tr, Y_tr
	ann.fit(X_tr,Y_tr,batch_size=512,epochs=50, validation_split=0.1)

	#=================
	matrix = PaCalC.all_partic_calib_curve(ann, X_te, Y_te, P_te)
	#=================
	# or single participant
	#=================
	# participants_dict = PaCalC.perParticipantDict(X_te, Y_te, P_te)
	# p_id = list(participants_dict.keys())[0]
	# matrix = PaCalC.partic_calib_curve(ann, *participants_dict[p_id])
	#=================

	print(matrix)

	if save:
		pickle.dump(matrix, open(f'out/PaCalC(dtst_seed={dtst_seed}).pkl','wb'))

	# either graph each indiv or graph avg


def PaCalC_p_cv(dtst_seed=214, p_cv=14, save=False):
	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1)

	ann = make_model(X_tr, Y_tr)

	# train model on X_tr, Y_tr
	ann.fit(X_tr,Y_tr,batch_size=512,epochs=50, validation_split=0.1)

	#=================
	matrix = PaCalC.all_partic_calib_curve(ann, X_te, Y_te, P_te, p_cv)

	print(matrix)
	if save:
		pickle.dump(matrix, open(f'out/PaCalC(dtst_seed={dtst_seed},p_cv={p_cv}).pkl','wb'))

	# either graph each indiv or graph avg


def PaCalC_dtst_cv(dtst_cv=4, p_cv=4, save=False):
	dtst_seeds = [randint(0,1000) for _ in range(0,cv)]

	out = []

	for dtst_seed in dtst_seeds:
		global _cached_Irregular_Surface_Dataset
		_cached_Irregular_Surface_Dataset=None

		X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(dtst_seed, True, split=0.1)

		ann = make_model(X_tr, Y_tr)

		# train model on X_tr, Y_tr
		ann.fit(X_tr,Y_tr,batch_size=512,epochs=50, validation_split=0.1)

		#=================
		matrix = PaCalC.all_partic_calib_curve(ann, X_te, Y_te, P_te, p_cv)

		print(matrix)
		out.append(matrix)

	if save:
		pickle.dump(PaCalC.pad_last_dim(out), open(f'out/PaCalC(dtst_cv={dtst_cv},p_cv={p_cv}).pkl','wb'))

	# either graph each indiv or graph avg

def make_model(X_tr, Y_tr):
	Lab = GL2G.data_processing()
	hid_layers=(606,303,606) #hidden layers
	model='classification' #problem type
	output= Y_tr.shape[-1] #ouput shape 
	input_shape=X_tr.shape[-1]
	ann=Lab.ANN(hid_layers=hid_layers,model=model,output=output,input_shape=input_shape,activation_hid='relu') # relu in hidden layers
	return ann

if __name__ == "__main__":
	PaCalC_F1(save=True)
	print('GREAT SUCCESS !')
	PaCalC_p_cv(save=True)
	print('GREAT SUCCESS !!')
	PaCalC_dtst_cv(save=True)
	print('GREAT SUCCESS !!!')