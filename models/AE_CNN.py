from .regrML import CAE, REGR_compile, REGR_pretrained_CNN

import tensorflow as tf
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam

def AE_CNN(x,y):
	print('In custom section')
	CAEx, Ex= CAE(x.shape[1:],None)

	CAEx = REGR_compile(CAEx, mse, Adam())

	print('Generated and compiled CAE')

	CAEx.summary()

	bs = 64
	# ep = 50
	ep = 5

	# CAEx = MLCC.train(CAEx,x,x,ep,bs,model_verbose,plot_verbose, cae_path)
	# ^ auto load endcoder?

	# subject_wise split needed?
	CAEx.fit(x,x, epochs=ep,
			batch_size=bs,
			validation_split=0.1
			)

	# with open(cae_path+'/'+'CAEx'+'_model_summary.txt', 'w') as f:
	# 	CAEx.summary(print_fn=lambda x: f.write(x + '\n'))

	# with open(cae_path+'/'+'Ex'+'_model_summary.txt', 'w') as f:
	# 	Ex.summary(print_fn=lambda x: f.write(x + '\n'))

	print('Trained encoder CAE')

	cnn = REGR_pretrained_CNN(Ex, y.shape[1:])

	print('Generated and compiled custom CNN')

	cnn.summary()

	return cnn