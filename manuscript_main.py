from load_data import load_surface_data, _CACHED_load_surface_data
from subject_wise_split import subject_wise_split

import numpy as np
from sklearn.metrics import f1_score

import copy
from random import seed, randint
from sklearn.metrics import classification_report
import tensorflow as tf

import manuscript_exp_func as mef

irreg_surfaces_labels = ['BnkL','BnkR', 'CS', 'FE', 'GR', 'SlpD', 'SlpU', 'StrD', 'StrU']

# function to repeat partic_calib_curve & avg_calib_curve over multiple seeds

if __name__ == "__main__":
	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(214, True, split=0.1)

	# train model on X_tr, Y_tr
	model = None

	#==========

	participants_dict = mef.perParticipantDict(X_te, Y_te, P_te)

	p_id = 15 # known id in P_te w/ seed
	print('P_id:', p_id)
	arr = mef.partic_calib_curve(model, *participants_dict[p_id])

	#=========

	# mef.avg_calib_curve(model, X_te, Y_te, P_te)