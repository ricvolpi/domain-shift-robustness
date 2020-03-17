import tensorflow as tf
import tensorflow.contrib.slim as slim

class Model(object):
    	"""Tensorflow model
    	"""
	def __init__(self, mode='train'):

		self.no_classes = 10
		self.img_size = 32
		self.no_channels = 3

	def encoder(self, images, reuse=False):

		with tf.variable_scope('encoder', reuse=reuse):
			with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
				with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):

					net = slim.conv2d(images, 64, 5, scope='conv1')
					net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
					net = slim.conv2d(net, 128, 5, scope='conv2')
					net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
					net = tf.contrib.layers.flatten(net)
					net = slim.fully_connected(net, 1024, scope='fc1')
					net = slim.fully_connected(net, 1024, scope='fc2')
					net = slim.fully_connected(net, self.no_classes, activation_fn=None, scope='fco')
					
					return net

	def build_model(self):

		# images placeholder
		self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.no_channels], 'images')
		# labels placeholder
		self.labels = tf.placeholder(tf.int64, [None], 'labels')
				
		self.logits = tf.squeeze(self.encoder(self.images))

		#for evaluation
		self.pred = tf.argmax(self.logits, 1)
		self.correct_pred = tf.equal(self.pred, self.labels)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		#variables for the minimizer are the net weights, variables for the maxmizer are the images' pixels
		min_vars = tf.trainable_variables()
				
		#loss for the minimizer
		self.min_loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)

		#we use Adam for the minimizer and vanilla gradient ascent for the maximizer 
		self.min_optimizer = tf.train.AdamOptimizer(self.learning_rate) 

		#minimizer
		self.min_train_op = slim.learning.create_train_op(self.min_loss, self.min_optimizer, variables_to_train = min_vars)

		min_loss_summary = tf.summary.scalar('min_loss', self.min_loss)

		accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
		self.summary_op = tf.summary.merge([min_loss_summary, accuracy_summary])		


