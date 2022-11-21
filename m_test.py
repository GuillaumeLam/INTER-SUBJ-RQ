import unittest

# for tests
import numpy as np
from load_data import load_surface_data, _CACHED_load_surface_data

# exported methods
from manuscript_exp_func import partic_calib_curve, avg_calib_curve, graph_calib_curve_per_Y, graph_calib

import manuscript_exp_func as PaCalC

# ==========
# dataset for tests

global _cached_Irregular_Surface_Dataset
_cached_Irregular_Surface_Dataset=None

import time

s = time.time()
X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(214, True, split=0.1)
e = time.time()

print('TIME TO HIT CACHE & SERVE:', e-s)

# ==========

class TDD_PaCalC(unittest.TestCase):

	def test_perParticipantDict(self):
		X, Y, P = np.array([[0]*100]*50), np.array([[1,0]]*25+[[0,1]]*25), np.array([1,2,3,4,5]*10)

		d = PaCalC.perParticipantDict(X, Y, P)

		for i,_id_ in enumerate(P):
			(x, y) = d[_id_]
			self.assertTrue((x == np.array([[0]*100]*10)).all())
			self.assertTrue((y == np.array([[1,0]]*5+[[0,1]]*5)).all())

	def test_perLabelDict(self):
		P_X, P_Y = np.array([[0]*100]*10), np.array([[1,0]]*5+[[0,1]]*5)

		d,m = PaCalC.perLabelDict(P_X, P_Y)

		self.assertEqual(m,5) # 10 cyles btwn 2 labels => 5 cycles per label

		for i, y in enumerate(P_Y):
			pos_y = y.argmax()
			p_x = d[pos_y]
			self.assertTrue((p_x == P_X[i]).all())

	def test_pad_curves(self):
		n_labels = 10
		f1_curves_per_label = []

		for i in range(1,n_labels+1):
			f1_curve = [(i)]*i
			i += 1
			f1_curves_per_label.append(f1_curve)

		F1 = PaCalC.pad_curves(f1_curves_per_label)

		self.assertEqual(F1.shape, (n_labels,n_labels))

class PaCalC_exported_func(unittest.TestCase):

	def test_partic_calib_curve(self):
		model = None
		P_X, P_Y = np.zeros((50,100)), np.array([[1,0]]*25+[[0,1]]*25) # replace None with 25 0's & 25 1's both ohe
		matrix = partic_calib_curve(model, P_X, P_Y)
		self.assertEqual(matrix.shape[0], 2) # check that output has right # of labels

	@unittest.expectedFailure
	def test_avg_calib_curve(self):
		model = None
		res = avg_calib_curve(model, X_te, Y_te, P_te)
		self.assertNotEqual(res, None)

	@unittest.expectedFailure
	def test_graph_calib_curve_per_Y(self):
		curve = None
		graph = graph_calib_curve_per_Y(curve)
		self.assertNotEqual(graph, None)

	@unittest.expectedFailure
	def test_graph_calib(self):
		curve = None
		graph = graph_calib(curve)
		self.assertNotEqual(graph, None)

if __name__ == '__main__':
	unittest.main()