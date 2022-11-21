import unittest

# data for tests
from load_data import load_surface_data, _CACHED_load_surface_data

# exported methods
from manuscript_exp_func import p_calib_curve, avg_calib_curve, graph_calib_curve_per_Y, graph_calib

# ==========
# dataset
global _cached_Irregular_Surface_Dataset
_cached_Irregular_Surface_Dataset=None

X_tr, Y_tr, P_tr, X_te, Y_te, P_te = _CACHED_load_surface_data(214, True, split=0.1)

# ==========

class TDD_internal_func(unittest.TestCase):

	def test_default(self):
		self.assertEqual('foo'.upper(), 'FOO')

class exported_func(unittest.TestCase):

	@unittest.expectedFailure
	def test_p_calib_curve(self):
		model = None
		P_X, P_Y = None, None
		res = p_calib_curve(model, P_X, P_Y)
		self.assertNotEqual(res, None)

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