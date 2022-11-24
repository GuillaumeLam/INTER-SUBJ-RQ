import unittest

# for tests
import numpy as np
from random import seed, randint
import tensorflow as tf
from tensorflow.keras.layers import Dense

from load_data import load_surface_data, _CACHED_load_surface_data

# exported methods
from manuscript_exp_func import partic_calib_curve, all_partic_calib_curve, calib_curve_cv, graph_calib_curve_per_Y, graph_calib

import manuscript_exp_func as PaCalC

# ==========
# dataset for tests

global _cached_Irregular_Surface_Dataset
_cached_Irregular_Surface_Dataset=None

import time

s = time.time()
X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(214, True, split=0.1)
e = time.time()

print('TIME TO HIT CACHE & SERVE:'+str(e-s)+'s')

seed(39)
np.random.seed(39)

# ==========

class PaCalC_exported_func(unittest.TestCase):

	def test_partic_calib_curve(self):
		
		matrix = partic_calib_curve(TestHelperFunc.make_model(), *TestHelperFunc.P_XY())
		
		self.assertEqual(matrix.shape, (2,6))

	def test_all_partic_calib_curve(self):		
		matrix = all_partic_calib_curve(TestHelperFunc.make_model(), *TestHelperFunc.XYP())
		
		self.assertEqual(matrix.shape, (5,2,2))

	def test_cv_single_partic(self):
		cv = 2
		
		matrix = calib_curve_cv(TestHelperFunc.make_model(), *TestHelperFunc.P_XY(), cv=cv)
		
		print(matrix.shape)

		self.assertEqual(matrix.shape, (cv,2,6))

	def test_cv_all_partic(self):
		cv = 2
		
		matrix = calib_curve_cv(TestHelperFunc.make_model(), *TestHelperFunc.XYP(), cv=cv)
		
		self.assertEqual(matrix.shape, (cv, 5, 2, 2))

	# @unittest.expectedFailure
	# def test_graph_calib_curve_per_Y(self):
	# 	curve = None
	# 	graph = graph_calib_curve_per_Y(curve)
	# 	self.assertNotEqual(graph, None)

	# @unittest.expectedFailure
	# def test_graph_calib(self):
	# 	curve = None
	# 	graph = graph_calib(curve)
	# 	self.assertNotEqual(graph, None)

class TDD_PaCalC(unittest.TestCase):

	def test_perLabelDict(self):
		P_X, P_Y = np.random.rand(10,100), np.array([[1,0]]*5+[[0,1]]*5)

		d,m = PaCalC.perLabelDict(P_X, P_Y)

		self.assertEqual(m,5) # 10 cyles btwn 2 labels => 5 cycles per label

		for i, (y, p_x) in enumerate(d.items()):
			pos_y = y.argmax()
			self.assertTrue((p_x == np.array(P_X[i*5:(i*5+5),:])).all())

	def test_pad_last_dim(self):
		n_labels = 10
		f1_curves_per_label = []

		for i in range(1,n_labels+1):
			f1_curve = [(i)]*i
			i += 1
			f1_curves_per_label.append(f1_curve)

		F1 = PaCalC.pad_last_dim(f1_curves_per_label)

		self.assertEqual(F1.shape, (n_labels,n_labels))

	def test_pad_last_dim_matrices(self):
		n_labels = 10
		f1_1 = []

		for i in range(1,n_labels+1):
			f1_curve = [(i)]*i
			i += 1
			f1_1.append(f1_curve)

		f1_2 = []

		for i in range(1,n_labels+1):
			f1_curve = [(i)]*(i+1)
			i += 1
			f1_2.append(f1_curve)

		f1_curves = [PaCalC.pad_last_dim(f1_1), PaCalC.pad_last_dim(f1_2)]

		F1 = PaCalC.pad_last_dim(f1_curves)

		self.assertEqual(F1.shape, (2,n_labels,n_labels+1))

	def test_perParticipantDict(self):
		X, Y, P = np.random.rand(50,100), np.array([[1,0]]*25+[[0,1]]*25), np.array([1,2,3,4,5]*10)

		d = PaCalC.perParticipantDict(X, Y, P)

		for i, (_id_, xy) in enumerate(d.items()):
			x, y = xy
			self.assertTrue((x == np.array(X[i::5,:])).all())
			self.assertTrue((y == np.array([[1,0]]*5+[[0,1]]*5)).all())

class TestHelperFunc:
	def make_model():
		model = tf.keras.models.Sequential()
		model.add(Dense(32, input_dim=100, activation='relu'))
		model.add(Dense(16, activation='relu'))
		model.add(Dense(2, activation='softmax'))

		return model

	def P_XY():
		P_X, P_Y = np.random.rand(50,100), np.array([[1,0]]*25+[[0,1]]*25) # replace None with 25 0's & 25 1's both ohe
		return P_X, P_Y

	def XYP():
		X, Y, P = np.random.rand(50,100), np.array([[1,0]]*25+[[0,1]]*25), np.array([1,2,3,4,5]*10)
		return X, Y, P

if __name__ == '__main__':
	unittest.main()