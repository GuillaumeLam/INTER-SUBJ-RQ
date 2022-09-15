import numpy as np
import tensorflow as tf

from load_data import load_surface_data


def weight_variable(shape):
	initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
	return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
	initializer = tf.constant_initializer(0.0)
	return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)


class IrregSurfaceAnn(object):
	def __init__(self,io_shape=(480,9), model_shape=(606,303,606)):
		ann = tf.keras.models.Sequential()
		ann.add(tf.keras.Input(shape=io_shape[0]))

		# fc1
		self.fc1 = tf.keras.layers.Dense(units=model_shape[0],activation='relu')
		ann.add(self.fc1)

		# fc2
		self.fc2 = tf.keras.layers.Dense(units=model_shape[1],activation='relu')
		ann.add(self.fc2)

		# fc3
		self.fc3 = tf.keras.layers.Dense(units=model_shape[2],activation='relu')
		ann.add(self.fc3)

		self.out = tf.keras.layers.Dense(units=output,activation='softmax')
		ann.add(self.out)

		ann.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

		self.layer = self.fc2.get_weights()[0]
		return ann

	def setWeights(self, session, weights):
		for v in tf.trainable_variables():
			session.run(v.assign(weights[v.name]))
   
		   
def train(args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, Ztrain):
    num_class = 9

    # x = tf.placeholder(tf.float32, (None, 28*28))
    # y = tf.placeholder(tf.float32, (None, num_class))
    # model = MNISTcnn(x, y, args)

    model = IrregSurfaceAnn()

    optimizer = tf.train.AdamOptimizer(1e-5).minimize(model.loss)

    saver = tf.train.Saver(tf.trainable_variables())

    weights = {}

    with tf.Session() as sess:
        print('Starting training')
        sess.run(tf.global_variables_initializer())
        if args.load_params:
            ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
            print('Restoring parameters from', ckpt_file)
            saver.restore(sess, ckpt_file)

        num_batches = Xtrain.shape[0] // args.batch_size
       
        validation = True
        val_num_batches = Xval.shape[0] // args.batch_size

        test_num_batches = Xtest.shape[0] // args.batch_size

        best_validate_accuracy = 0
        score = 0

        # Phase One
        for epoch in range(args.epochs):
            begin = time.time()

            # train
            train_accuracies = []
            for i in range(num_batches):

                batch_x = Xtrain[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_y = Ytrain[i*args.batch_size:(i+1)*args.batch_size,:]

                _, acc = sess.run([optimizer, model.accuracy], feed_dict={x: batch_x, y: batch_y})
                train_accuracies.append(acc)
            train_acc_mean = np.mean(train_accuracies)

            # compute loss over validation data
            if validation:
                val_accuracies = []
                for i in range(val_num_batches):
                    batch_x = Xval[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = Yval[i*args.batch_size:(i+1)*args.batch_size,:]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)

                # log progress to console
                print("Epoch %d, time = %ds, train accuracy = %.4f, validation accuracy = %.4f" % (
                epoch, time.time() - begin, train_acc_mean, val_acc_mean))
            else:
                print("Epoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time() - begin, train_acc_mean))
            sys.stdout.flush()

            if val_acc_mean > best_validate_accuracy:
                best_validate_accuracy = val_acc_mean

                test_accuracies = []
                for i in range(test_num_batches):
                    batch_x = Xtest[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = Ytest[i*args.batch_size:(i+1)*args.batch_size,:]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                    test_accuracies.append(acc)
                score = np.mean(test_accuracies)

                print("Best Validated Model Prediction Accuracy = %.4f " % (score))

            if (epoch + 1) % 10 == 0:
                ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
                saver.save(sess, ckpt_file)

        for v in tf.trainable_variables():
            weights[v.name] = v.eval()

        # Phase Two
        weight_pre = weights[args.layer]
        changes = np.zeros_like(weights)
        for epoch in range(args.epochs_cf):
            begin = time.time()

            # train
            for i in range(num_batches):

                batch_x = Xtrain[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_z = Ztrain[i*args.batch_size:(i+1)*args.batch_size,:]

                _, acc, weight = sess.run([optimizer, model.accuracy, model.layer], feed_dict={x: batch_x, y: batch_z})
                changes += np.abs(weight - weight_pre)/np.max(np.abs(weight - weight_pre))
                weight_pre = weight
        changes = changes/(args.epochs_cf*num_batches)


        # Phase Three
        weights[args.layer][changes>args.threshold] = 0
        model.setWeights(sess, weights)

        ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
        saver.save(sess, ckpt_file)


if __name__ == "__main__":
	seed = 39

	X_tr, Y_tr, P_tr, X_te, Y_te, P_te, _ = load_surface_data(seed, True, split=0.1)

	X_tr, Y_tr, X_val, Y_val, P_tr, P_val = subject_wise_split(X_tr, Y_tr, P_tr, split=0.1, seed=seed, subject_wise=True)

	tf.set_random_seed(seed)
	np.random.seed(seed)

	train(X_tr, Y_tr, X_val, Y_val, X_te, Y_te, P_tr)