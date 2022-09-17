import argparse
import numpy as np
import os.path
from os import path
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K

from random import seed, randint
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

from load_data import load_surface_data


def weight_variable(shape):
	initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
	return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
	initializer = tf.constant_initializer(0.0)
	return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def recall_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))


class ModelPlus(Model):
	def __init__(self, inputs, outputs):
		super().__init__(inputs, outputs)

	def getWeights(self, layers=['fc2']):
		return [np.array(layer.get_weights()[0]) if (layer.name in layers) else None for layer in self.layers]

	def setWeights(self, weights):
		for (layer, weight) in zip(self.layers, weights):
			if weight is not None:
				_, bias = layer.get_weights()
				layer.set_weights([weight, bias])

	def freezeOtherWeights(self, layers=['fc2']):
		for layer in self.layers:
			if layer.name not in layers:
				layer.trainable = False


class CF_DL(object):
	def __init__(self, z_shape, x_shape=(480), y_shape=9, model_shape=(606,303,606)):
		inputs = Input(shape=x_shape)
		fc1 = Dense(model_shape[0],activation='relu', name='fc1')(inputs)
		fc2 = Dense(model_shape[1],activation='relu', name='fc2')(fc1)
		fc3 = Dense(model_shape[2],activation='relu', name='fc3')(fc2)

		out_y = Dense(y_shape,activation='softmax', name='out_y')(fc3)

		out_z = Dense(z_shape,activation='softmax', name='out_z')(fc3)

		self.model_y = ModelPlus(inputs=inputs, outputs=out_y)
		self.model_y._name = 'IS_Ann_Y'
		self.model_y.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc', f1_m])

		self.model_z = ModelPlus(inputs=inputs, outputs=out_z)
		self.model_z._name = 'IS_Ann_Z'
		self.model_z.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc', f1_m])

		self.setMode('model_y')

		self.model_y.summary()
		self.model_z.summary()

	def setMode(self, mode):
		self.ann = getattr(self, mode)
		self.mode = mode
		print('Switched to ', mode)

	def switchMode(self):
		if self.mode == 'model_y':
			self.setMode('model_z')
		else:
			self.setMode('model_y')


class WeightTrackerCallback(tf.keras.callbacks.Callback):
	def __init__(self, weights_pre, changes, layers):
		self.weight_pre = weights_pre
		self.changes = changes
		self.layers = layers

	def on_train_batch_end(self, batch, logs=None):
		ann = self.model
		weight = ann.getWeights(layers=self.layers)

		changes = [np.abs(np.array(w) - np.array(w_pre))/np.max(np.abs(np.array(w) - np.array(w_pre))) if w_pre is not None else None for w, w_pre in zip(weight, self.weight_pre)]
		
		for i,c in enumerate(changes):
			if c is not None:
				self.changes[i] += c
		self.weight_pre = weight


def CF(args, Xtrain, Ytrain, Xtest, Ytest, Ptrain):
	num_class = 9

	z_shape = len(set(Ptrain))

	IS_CF_DL = CF_DL(z_shape)

	# Phase One
	print('='*30)
	print('Phase One')
	print('1'*30)
	print('='*30)

	history = IS_CF_DL.ann.fit(
		Xtrain,
		Ytrain,
		epochs=args.epochs,
		batch_size=args.batch_size,
		validation_split=0.1,
		callbacks=[
			tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min",restore_best_weights=True)
		]
	)
	weights = IS_CF_DL.ann.getWeights(layers=args.layer)

	# Phase Two
	print('='*30)
	print('Phase Two')
	print('2'*30)
	print('='*30)

	weight_pre = IS_CF_DL.ann.getWeights(layers=args.layer)

	def hof_None_check(func):
		def hof(i):
			if i is not None:
				return func(i)
		return hof

	changes = list(map(hof_None_check(np.zeros_like), weight_pre))

	esc = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min",restore_best_weights=True)
	wtc = WeightTrackerCallback(weight_pre, changes, args.layer)

	IS_CF_DL.switchMode()
	IS_CF_DL.ann.freezeOtherWeights(layers=args.layer)

	ohe = OneHotEncoder()
	Ptrain_oh = ohe.fit_transform(Ptrain.reshape(-1, 1)).toarray()

	history = IS_CF_DL.ann.fit(
		Xtrain,
		Ptrain_oh,
		epochs=args.epochs_cf,
		batch_size=args.batch_size,
		validation_split=0.1,
		callbacks=[
			esc,
			wtc
		]
	)
	num_batches = Xtrain.shape[0]//args.batch_size
	norm_func = lambda x: x/(esc.stopped_epoch*num_batches)
	norm_changes = list(map(hof_None_check(norm_func), wtc.changes))

	# Phase Three
	print('='*30)
	print('Phase Three')
	print('3'*30)
	print('='*30)

	num_neuron_pruned = 0

	for i, nc in enumerate(norm_changes):
		if weights[i] is not None and nc is not None:
			# check distribution of weights

			# todo: given percentage, change threshold => easier to check on % of pruning
			indices = np.where(nc > args.threshold)[0]
			num_neuron_pruned += indices.shape[0]
			print(f"Pruned {indices.shape[0]/(weights[i].shape[0]*weights[i].shape[1])*100}% of layer {i}")
			weights[i].put(indices, 0)


	IS_CF_DL.ann.setWeights(weights)
	# save model

	# Testing Phase
	IS_CF_DL.switchMode()
	Yhat = IS_CF_DL.ann.predict(Xtest)
	f1_score = f1_m(np.array(Ytest, dtype="float32"), np.array(Yhat, dtype="float32")).numpy()

	return IS_CF_DL, f1_score

def cv_hparam(args):
	seed(39)
	seeds = [randint(0,1000) for _ in range(0,args.cv_folds)]

	if args.seed is not None:
		print('Producing model with given seed only!')
		seeds = [args.seed]

	h_param = {'threshold':[0.005, 0.01, 0.05], 'layer': [['fc1'],['fc2'],['fc3'],['fc1','fc2'],['fc1','fc3'],['fc2','fc3'],['fc1','fc2','fc3']]}

	def cross(h_param):
		comb = []
		for k,v in [(k,v) for k,v in h_param.items()]:
			if len(comb)==0:
				for arg_val in v:
					comb.append({k:arg_val})
			else:
				cross = comb
				comb = []

				for d in cross:
					for arg_val in v:
						new_d = dict(d)
						new_d[k]=arg_val 
						comb.append(new_d)
		return comb

	h_param = cross(h_param)

	def get_hparam(args):
		return f"threshold={args.threshold},layer={args.layer}"

	scores = {}
	models = {}
	if path.exists('CF_scores.pkl'):
		scores = pickle.load(open('CF_scores.pkl', 'rb'))

	for d in h_param:
		args.threshold=d['threshold']
		args.layer=d['layer']

		if get_hparam(args) not in scores:
			scores[get_hparam(args)]={}
		models[get_hparam(args)]={}

		for s in seeds:
			if s in scores[get_hparam(args)] and args.seed is None:
				s = randint(0,1000)

			X_tr, Y_tr, P_tr, X_te, Y_te, P_te, _ = load_surface_data(s, True, split=0.1)

			tf.random.set_seed(s)
			np.random.seed(s)

			IS_CF_DL, f1_score = CF(args, X_tr, Y_tr, X_te, Y_te, P_tr)
			
			scores[get_hparam(args)][s]=f1_score
			models[get_hparam(args)][s]=IS_CF_DL

			pickle.dump(scores, open('CF_scores.pkl','wb'))

			print(scores)

	return scores, models

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--epochs', type=int, default=50, help='How many epochs to run in first phase?')
	parser.add_argument('-p', '--epochs_cf', type=int, default=50, help='How many epochs to run in second phase?')
	parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size during training per GPU')
	parser.add_argument('-t', '--threshold', type=float, default=0.01, help='threshold of updating the weights')
	parser.add_argument('-f', '--cv_folds', type=int, default=7, help='number of cross validation folds')
	parser.add_argument('-s', '--seed', type=int, default=None, help='seed')
	parser.add_argument('-n', '--layer', nargs='+', default=['fc1', 'fc2', 'fc3'], help='the layer we are interested in adjust')
	args = parser.parse_args()

	scores, models = cv_hparam(args)

	# scores are automatically saved, model can be replicated with hparams and layer selection