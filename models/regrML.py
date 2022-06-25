import functools
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout, BatchNormalization

from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, Cropping2D

from tensorflow.keras.utils import plot_model

def REGR_model(model_id, in_shape, out_shape, verbose=False):

	model = Sequential()
	
	if model_id == 'CAE+DENSE':
		# C1
		model.add(Conv2D(16, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', input_shape=in_shape, padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		# C2
		model.add(Conv2D(32, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		#C3
		model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		#C4
		model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		#D1
		model.add(Conv2DTranspose(128, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(UpSampling2D(size=(2,2)))

		model.add(Conv2DTranspose(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(UpSampling2D(size=(2,2)))

		model.add(Conv2DTranspose(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(UpSampling2D(size=(2,2)))

		model.add(Conv2DTranspose(16, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(UpSampling2D(size=(2,2)))

		model.add(Conv2DTranspose(3, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(Flatten())
		# model.add(Dense(functools.reduce(lambda a, b: a*b,out_shape)))
		# model.add(Reshape(out_shape))

	elif model_id in ['CAE_L_SYM','CAE_M', 'CAE_S']:
		model.add(Input(shape=in_shape))

		model.add(ZeroPadding2D(padding=((2,1),(1,0))))

		# C1
		model.add(Conv2D(16, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		# C2
		model.add(Conv2D(32, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		#C3
		model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		ch_out = 128

		#C4
		model.add(Conv2D(ch_out, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))
		
		ch_in = 128
			
		if model_id == 'CAE_M':
			latent = (7,1,128)
			# latent_out = (26,2,16)

			model.add(Flatten())
			model.add(Dense(functools.reduce(lambda a, b: a*b, latent)))
			# model.add(Dense(512))
			model.add(Dense(functools.reduce(lambda a, b: a*b, latent)))
			model.add(Reshape(latent))
		
		elif model_id == 'CAE_S':
			latent = (7,1,128)
			# latent_out = (26,2,16)

			model.add(Flatten())
			model.add(Dense(functools.reduce(lambda a, b: a*b, latent)))
			model.add(Reshape(latent))

		elif model_id == 'CAE_L_SYM':
			model.add(Dense(ch_out))
			model.add(Dense(1024))
			model.add(Dense(4096))
			model.add(Dense(1024))
			model.add(Dense(ch_in))

		#D1
		model.add(Conv2DTranspose(ch_in, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(UpSampling2D(size=(2,2)))

		model.add(Conv2DTranspose(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(UpSampling2D(size=(2,2)))

		model.add(Conv2DTranspose(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(UpSampling2D(size=(2,2)))

		model.add(Conv2DTranspose(16, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(UpSampling2D(size=(2,2)))

		model.add(Conv2DTranspose(3, kernel_size=(3,3), kernel_initializer='he_uniform', padding='same'))

		model.add(Cropping2D(cropping=((6,5),(5,5))))

	elif model_id in ['CNN_S','CNN_M','CNN_L']:
		model.add(Input(shape=in_shape))

		base_ch = 16

		# C1
		model.add(Conv2D(base_ch, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', input_shape=in_shape, padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		# C2
		model.add(Conv2D(base_ch*2, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		#C3
		model.add(Conv2D(base_ch*4, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		#C4
		model.add(Conv2D(base_ch*16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
		model.add(BatchNormalization(center=True, scale=True))
		model.add(Dropout(0.5))

		model.add(Flatten())

		if model_id == 'CNN_S':
			model.add(Dense(1024))
		elif model_id == 'CNN_M':
			model.add(Dense(1024))
			model.add(Dense(1024))
		elif model_id == 'CNN_L':
			model.add(Dense(2048))
			model.add(Dense(1024))
		
		# model.add(Dense(functools.reduce(lambda a, b: a*b,out_shape)))
		# model.add(Reshape(out_shape))

	# simple FFN
	elif model_id == 'FFN_flat':
		
		model.add(Input(shape=in_shape))

		model.add(Flatten())
		model.add(Dense(functools.reduce(lambda a, b: a*b, in_shape)))

		model.add(Dense(256))

		# model.add(Dense(functools.reduce(lambda a, b: a*b, out_shape)))
		# model.add(Reshape(out_shape))

	# ~size of CAE_L
	elif model_id == 'FFN_rect':

		model.add(Input(shape=in_shape))

		model.add(Flatten())

		model.add(Dense(2048))

		# model.add(Dense(functools.reduce(lambda a, b: a*b, out_shape)))
		# model.add(Reshape(out_shape))

	model.add(Flatten())
	model.add(Dense(out_shape[0], activation='softmax'))

	verbose and model.summary()
	# verbose and plot_model(model, to_file='AE-nw.png')
	return model

def REGR_compile(m, loss_fnc, opt='rmsprop'):
	m.compile(optimizer=opt, loss=loss_fnc, metrics =["accuracy"])
	return m	

def CAE(d_shape, latent):
	# d_shape => ex. X.shape[1:]=(101,7/6,6/3) ie. shape of data without # of examples 

	ch = d_shape[-1]

	# better reconstruction of data, exp incr and decr of ch
	if d_shape[1] == 7:
		in_pad = ((6,5),(5,4))
		out_crp = ((6,5),(5,4))
		base = 16
		# 32 too low, 64
	elif d_shape[1] == 6:
		in_pad = ((2,1),(1,1))
		out_crp = ((6,5),(5,5))
		base = 16

	input_e = Input(shape=d_shape)
	e = ZeroPadding2D(padding=in_pad)(input_e)
	e = conv_bloc_l(base,e)
	e = conv_bloc_l(base*2,e)
	e = conv_bloc_l(base*4,e)
	e = conv_bloc_l(base*16,e)
	Ed = Model(input_e, e)

	print('^'*25)
	print('!'*25)
	print('Encoder:')
	Ed.summary()

	input_d = Input(shape=tuple(e.shape[1:].as_list()))
	d = dconv_bloc_l(base*16,input_d)
	d = dconv_bloc_l(base*4,d)
	d = dconv_bloc_l(base*2,d)
	d = dconv_bloc_l(base,d)
	d = Conv2DTranspose(ch, kernel_size=(3,3), kernel_initializer='he_uniform', padding='same')(d)
	d = Cropping2D(cropping=out_crp)(d)

	Dd = Model(input_d, d)

	print('-'*25)
	print('latent dim:' + str(functools.reduce(lambda a, b: a*b, e.shape[1:].as_list())) )
	print('-'*25)
	print('Decoder:')
	Dd.summary()
	print('!'*25)
	print('V'*25)

	pred = Dd(Ed(input_e))
	CAE = Model(input_e, pred)
	print('CAE for data done!')
	return CAE, Ed

# def conv_bloc_m(units, model):
# 	model.add(Conv2D(units, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', padding='same'))
# 	model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
# 	model.add(BatchNormalization(center=True, scale=True))
# 	model.add(Dropout(0.5))

def conv_bloc_l(units, layer):
	c = Conv2D(units, kernel_size=(3,3), activation='relu',kernel_initializer='he_uniform', padding='same')(layer)
	maxp = MaxPooling2D(pool_size=(2,2), padding='same')(c)
	bn = BatchNormalization(center=True, scale=True)(maxp)
	conv_bloc = Dropout(0.5)(bn)

	return conv_bloc

def dconv_bloc_l(units, layer):
	d = Conv2DTranspose(units, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(layer)
	dconv_bloc = UpSampling2D(size=(2,2))(d)

	return dconv_bloc

def REGR_pretrained_CNN(Ex,out_shape,save_path=None):
	# freeze Ex
	for layer in Ex.layers:
		layer.trainable = False	

	# make large model
	# latent = (7,1,128)
	input_d = Input(shape=Ex.layers[-1].output_shape[1:])

	d = Flatten()(input_d)
	d = Dense(1024)(d)
	d = Dense(1024)(d)
	d = Dense(functools.reduce(lambda a, b: a*b, out_shape))(d)
	d = Reshape(out_shape)(d)

	dense = Model(input_d, d)

	input_e = Input(shape=Ex.layers[0].input_shape[0][1:])

	mid = Ex(input_e)
	out = dense(mid)
	CNN = Model(input_e, out)

	if save_path is not None:
		with open(save_path+'/'+'Dense'+'_model_summary.txt', 'w') as f:
			dense.summary(print_fn=lambda x: f.write(x + '\n'))

	return CNN