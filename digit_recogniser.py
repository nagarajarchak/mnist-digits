import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnistdigits = input_data.read_data_sets('MNIST_data', one_hot=True)

print('Loaded MNIST Dataset.')

xtrain = mnistdigits.train.images
ytrain = mnistdigits.train.labels
xdev = mnistdigits.validation.images
ydev = mnistdigits.validation.labels
xtest = mnistdigits.test.images
ytest = mnistdigits.test.labels

m = xtrain.shape[0]  # Number of training examples.
n_iter = 1001        # Number of iterations.

with tf.name_scope("Input"):
	X = tf.placeholder(tf.float32, name = "X", shape=[None,784])
	X_Reshape = tf.reshape(X,[-1, 28, 28, 1])
	tf.summary.image('Input',X_Reshape,3)
	Y = tf.placeholder(tf.float32, name = "Y", shape=[None, 10])

with tf.name_scope("Weights & Biases"):
	W1 = tf.get_variable("W1", [784,100], initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [1,100], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [100,100], initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [1,100], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [100,10], initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [1,10], initializer = tf.zeros_initializer())

def neuralnetForwardProp(inputData):

	with tf.name_scope("HLayer1"):
		z1 = tf.add(tf.matmul(inputData,W1), b1)
		a1 = tf.nn.relu(z1)
		tf.summary.histogram("Weight1",W1)

	with tf.name_scope("HLayer2"):
		z2 = tf.add(tf.matmul(a1,W2), b2)
		a2 = tf.nn.relu(z2)
		tf.summary.histogram("Weight2",W2)

	with tf.name_scope("HLayer3"):
		z3 = tf.add(tf.matmul(a2,W3), b3)
		tf.summary.histogram("Weight3",W3)
	
	return z3

def neuralnetBackProp(inputData):

	predict = neuralnetForwardProp(inputData)

	with tf.name_scope("Cost"):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict, labels = Y)) # Compute cost.
	with tf.name_scope("Optimizer"):
		gradOptimizer = tf.train.RMSPropOptimizer(learning_rate = 0.001).minimize(cost)	# Using different optimizer.

	tf.summary.scalar("Cross Entropy",cost)

	numepochs = 50     # Number of Epochs
	batchsize = 500    # Batch Size
 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())  # Initialize all global variables.
		merged_summary = tf.summary.merge_all()
		writer = tf.summary.FileWriter("tmp/DeepLearning/26")
		writer.add_graph(sess.graph)
		for epochs in range(numepochs):
			epochloss = 0
			for _ in range(int(m/batchsize)):
				epochX, epochY = mnistdigits.train.next_batch(batchsize)
				_, c = sess.run([gradOptimizer,cost], feed_dict={X: epochX, Y: epochY})
				summary = sess.run(merged_summary, feed_dict={X: epochX, Y: epochY})
				writer.add_summary(summary, numepochs)
				epochloss += c
			print('Epoch', epochs,'completed out of', numepochs,' Loss:',epochloss)

		correctPredict = tf.equal(tf.argmax(predict,1),tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(correctPredict, tf.float32))

		trainAccu = accuracy.eval({X: xtrain, Y: ytrain})
		devAccu = accuracy.eval({X: xdev, Y: ydev})
		testAccu = accuracy.eval({X: xtest, Y: ytest})

	print("Train Dataset Accuracy: %f" %(trainAccu))
	print("Development Dataset Accuracy: %f" %(devAccu))
	print("Test Dataset Accuracy: %f" %(testAccu))

neuralnetBackProp(X)	