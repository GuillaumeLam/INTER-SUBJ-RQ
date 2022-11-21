from load_data import load_surface_data, _CACHED_load_surface_data
from subject_wise_split import subject_wise_split
import GaitLab2Go as GL2G

import numpy as np
from sklearn.metrics import f1_score

import copy
from random import seed, randint
from sklearn.metrics import classification_report
import tensorflow as tf

import manuscript_exp_func as mef

irreg_surfaces_labels = ['BnkL','BnkR', 'CS', 'FE', 'GR', 'SlpD', 'SlpU', 'StrD', 'StrU']

# function to repeat partic_calib_curve & avg_calib_curve over multiple seeds

def f1_vs_C_tr(seed=214):
	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(seed, True, split=0.1)

	Lab = GL2G.data_processing()
	hid_layers=(606,303,606) #hidden layers
	model='classification' #problem type
	output= Y_tr.shape[-1] #ouput shape 
	input_shape=X_tr.shape[-1]
	ann=Lab.ANN(hid_layers=hid_layers,model=model,output=output,input_shape=input_shape,activation_hid='relu') # relu in hidden layers

	# train model on X_tr, Y_tr
	ann.fit(X_tr,Y_tr,batch_size=512,epochs=50, validation_split=0.1)

	matrix = mef.avg_calib_curve(ann, X_te, Y_te, P_te)

	print(matrix)

	# either graph each indiv or graph avg

if __name__ == "__main__":

	f1_vs_C_tr()