from load_data import load_surface_data, _CACHED_load_surface_data
from subject_wise_split import subject_wise_split

import numpy as np
from sklearn.metrics import f1_score

import copy
from random import seed, randint
from sklearn.metrics import classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

import manuscript_exp_func as mef

if __name__ == "__main__":
	global _cached_Irregular_Surface_Dataset
	_cached_Irregular_Surface_Dataset=None

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(214, True, split=0.1)

	mef.per_label_calib(None, (X_te, Y_te, P_te))