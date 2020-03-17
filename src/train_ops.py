import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
from ConfigParser import *
import os
import cPickle
import scipy.io
import sys
import glob
from numpy.linalg import norm
from scipy import misc
import skimage.transform

import PIL.Image

sys.path.insert(0,'../')
from transformation_ops import TransfOps
from search_ops import SearchOps

class TrainOps(object):

	def __init__(self, model, exp_dir):

		self.model = model
		self.exp_dir = exp_dir

		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth=False

		self.data_dir = './data'

	def load_exp_config(self):

		print self.exp_dir

		config = ConfigParser()

		print 'LOADING CONFIG FILE'
		config.read(os.path.join(self.exp_dir,'exp_config'))
		self.source_dataset = config.get('EXPERIMENT_SETTINGS', 'source_dataset')

		self.model.source_dataset = self.source_dataset
			
		self.model.no_classes = 10
		self.model.img_size = 32

		self.log_dir = os.path.join(self.exp_dir,'logs')
		self.model_save_path = os.path.join(self.exp_dir,'model')
		self.images_dir = os.path.join(self.exp_dir,'images')

		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		if not os.path.exists(self.model_save_path):
			os.makedirs(self.model_save_path)

		if not os.path.exists(os.path.join(self.images_dir)):
			os.makedirs(os.path.join(self.images_dir))


		self.train_iters = config.getint('MAIN_SETTINGS', 'train_iters')
		self.batch_size = config.getint('MAIN_SETTINGS', 'batch_size')
		self.model.batch_size = self.batch_size
		self.model.learning_rate = config.getfloat('MAIN_SETTINGS', 'learning_rate')

		self.transf_string = config.get('MAIN_SETTINGS', 'transf_string')
		self.sub_train_iters = config.getint('MAIN_SETTINGS', 'sub_train_iters')
		self.string_length = config.getint('MAIN_SETTINGS', 'string_length')

		self.transf_ops = TransfOps()
		self.search_ops = SearchOps()

	def load_svhn(self, split='train'):

		print ('Loading SVHN dataset.')

		image_file = 'train_32x32.mat' if split=='train' else 'test_32x32.mat'

		image_dir = os.path.join(self.data_dir, 'svhn', image_file)
		svhn = scipy.io.loadmat(image_dir)
		images = np.transpose(svhn['X'], [3, 0, 1, 2])
		labels = svhn['y'].reshape(-1)
		labels[np.where(labels==10)] = 0
		images = images/255.
		return images, labels

	def load_mnist(self, split='train'):

		print ('Loading MNIST dataset.')
		image_file = 'train.pkl' if split=='train' else 'test.pkl'
		image_dir = os.path.join(self.data_dir, 'mnist', image_file)
		with open(image_dir, 'rb') as f:
			mnist = cPickle.load(f)
		
		images = mnist['X'] 
		labels = mnist['y']

		images = images
		images = images/255. # better generalization performance if [0,1]

		images = np.stack((images,images,images), axis=3) # grayscale to rgb

		return np.squeeze(images), labels

	def load_mnist_m(self, split='train'):

		print ('Loading MNIST_M dataset.')


		image_dir = os.path.join(self.data_dir,'mnist_m')

		if split == 'train':
			data_dir = os.path.join(image_dir,'mnist_m_train')
			with open(os.path.join(image_dir,'mnist_m_train_labels.txt')) as f:
				content = f.readlines()
				
		elif split == 'test':
			data_dir = os.path.join(image_dir,'mnist_m_test')
			with open(os.path.join(image_dir,'mnist_m_test_labels.txt')) as f:
				content = f.readlines()


		content = [c.split('\n')[0] for c in content]
		images_files = [c.split(' ')[0] for c in content]
		labels = np.array([int(c.split(' ')[1]) for c in content]).reshape(-1)

		images = np.zeros((len(labels), 32, 32, 3))

		for no_img,img in enumerate(images_files):
			img_dir = os.path.join(data_dir, img)
			im = misc.imread(img_dir)
			im = np.expand_dims(im, axis=0)
			images[no_img] = im

		images = images 
		images = images/255.
		
		return images, labels

	def load_syn(self, split='train'):
		print ('Loading SYN dataset.')

		image_file = 'synth_train_32x32.mat' if split=='train' else 'synth_test_32x32.mat'

		image_dir = os.path.join(self.data_dir,'syn', image_file)
		syn = scipy.io.loadmat(image_dir)
		images = np.transpose(syn['X'], [3, 0, 1, 2])
		labels = syn['y'].reshape(-1)
		labels[np.where(labels==10)] = 0
		
		images = images/255.
		return images, labels

	def load_usps(self, split='train'):

		print ('Loading USPS dataset.')
		image_file = 'usps_32x32.pkl'
		image_dir = os.path.join(self.data_dir,'usps', image_file)
		
		with open(image_dir, 'rb') as f:
			usps = cPickle.load(f)
		
		images = usps['X']
		labels = usps['y']
		labels -= 1
		labels[labels==255] = 9

		images=np.squeeze(images)
		images = np.stack((images,images,images), axis=3) # grayscale to rgb
		images = images/255.

		if split == 'train':
			return images[:6562], np.squeeze(labels[:6562]).astype(int)
		elif split == 'validation':
			return images[6562:7291], np.squeeze(labels[6562:7291]).astype(int)
		elif split == 'test':	    
			return images[7291:], np.squeeze(labels[7291:]).astype(int)
	
	def load_test_data(self, target):

		if target=='mnist_m':
			self.target_test_images, self.target_test_labels = self.load_mnist_m(split='test')
		elif target=='svhn':
			self.target_test_images, self.target_test_labels = self.load_svhn(split='test')
		elif target=='syn':
			self.target_test_images, self.target_test_labels = self.load_syn(split='test')
		elif target=='usps':
			self.target_test_images, self.target_test_labels = self.load_usps(split='test')
		elif target=='mnist':
			self.target_test_images, self.target_test_labels = self.load_mnist(split='test')

		return self.target_test_images,self.target_test_labels

	def train(self, random_transf=False): 

		'''
		This method allows to train ERM and RDA models.
		
			random_transf: if set to True, RDA is used, o.w. ERM.
		
		The number of transformations to be concatenated needs be to
		set in the file exp_config.
		'''

		# build a graph
		print 'Building model'
		self.model.build_model()
		print 'Built'

		print 'Loading data'

		source_train_images, source_train_labels = self.load_mnist(split='train')
		target_test_images, target_test_labels = self.load_mnist(split='test')

		with tf.Session(config=self.config) as sess:
			tf.global_variables_initializer().run()

			saver = tf.train.Saver()

			summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

			print 'Training'
			
			for t in range(self.train_iters):

				i = t % int(source_train_images.shape[0] / self.batch_size)

				#current batch of images and labels
				batch_images = source_train_images[i*self.batch_size:(i+1)*self.batch_size]
				batch_labels = source_train_labels[i*self.batch_size:(i+1)*self.batch_size]

				if random_transf:
					batch_images, _, _ = self.transf_ops.transform_dataset(batch_images * 255., transf_string = self.transf_string)
					batch_images /= 255.
				
				feed_dict = {self.model.images: batch_images, self.model.labels: batch_labels} 

				#running a step of gradient descent
				sess.run([self.model.min_train_op, self.model.min_loss], feed_dict) 

				#evaluating the model
				if t % 2500 == 0:

					summary, min_l, acc = sess.run([self.model.summary_op, self.model.min_loss, self.model.accuracy], feed_dict)

					train_rand_idxs = np.random.permutation(source_train_images.shape[0])[:100]
					test_rand_idxs = np.random.permutation(target_test_images.shape[0])[:100]

					train_acc, train_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
					feed_dict={self.model.images: source_train_images[train_rand_idxs], 
					self.model.labels: source_train_labels[train_rand_idxs]})
					
					test_acc, test_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
					feed_dict={self.model.images: target_test_images[test_rand_idxs], 
					self.model.labels: target_test_labels[test_rand_idxs]})
					  
					summary_writer.add_summary(summary, t)
					print ('Step: [%d/%d] train_min_loss: [%.4f] train_acc: [%.4f] test_min_loss: [%.4f] test_acc: [%.4f]'%(t+1, self.train_iters, train_min_loss, train_acc, test_min_loss, test_acc))
			
				if t % 10000 == 0:
					print 'Saving'
					saver.save(sess, os.path.join(self.model_save_path, 'encoder'))

	def train_search(self, search_algorithm='random_search'): 
		
		'''
		This method allows to train models using RSDA and ESDA algorithms.
		Referring to the paper, this is Algorithm 3.

			search_algorithm: 'random_search' or 'evolution_search', 
							  accordingly to the desired search procedure.

		The number of transformations to be concatenated needs be to
		set in the file exp_config.
		'''

		# build a graph
		print 'Building model'
		self.model.build_model()
		print 'Built'

		print 'Loading data'

		source_train_images, source_train_labels = self.load_mnist(split='train')
		target_test_images, target_test_labels = self.load_mnist(split='test')

		# initializing the set of data augmentation rules.

		transformations = [['identity']]
		levels = [[None]]		

		with tf.Session(config=self.config) as sess:

			tf.global_variables_initializer().run()

			saver = tf.train.Saver()

			summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

			print 'Training'
			
			for t in range(self.train_iters):

				i = t % int(source_train_images.shape[0] / self.batch_size)

				# current batch of images and labels
				batch_images = source_train_images[i*self.batch_size:(i+1)*self.batch_size]
				batch_labels = source_train_labels[i*self.batch_size:(i+1)*self.batch_size]

				# sampling uniformly a transformation and its level, and applying it to the batch
				rnd_transf_idx = npr.randint(len(transformations))
				
				if transformations[rnd_transf_idx] == ['identity']: # do nothing for 'identity', namely use original images
					pass
				else:
					# TransfOps requires [0,255] pixel ranges, while here images [0,1]
					batch_images, _, _ = self.transf_ops.transform_dataset(batch_images * 255., transformations=transformations[rnd_transf_idx], levels=levels[rnd_transf_idx])
					batch_images /= 255.
								
				# running a step of gradient descent
				feed_dict = {self.model.images: batch_images, self.model.labels: batch_labels} 
				sess.run([self.model.min_train_op, self.model.min_loss], feed_dict) 

				#evaluating the model
				if t % 2500 == 0:

					summary, min_l, acc = sess.run([self.model.summary_op, self.model.min_loss, self.model.accuracy], feed_dict)

					train_rand_idxs = np.random.permutation(source_train_images.shape[0])[:100]
					test_rand_idxs = np.random.permutation(target_test_images.shape[0])[:100]

					train_acc, train_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
					feed_dict={self.model.images: source_train_images[train_rand_idxs], 
					self.model.labels: source_train_labels[train_rand_idxs]})
					test_acc, test_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
					feed_dict={self.model.images: target_test_images[test_rand_idxs], 
					self.model.labels: target_test_labels[test_rand_idxs]})
					  
					summary_writer.add_summary(summary, t)
					print('Step: [%d/%d] train_min_loss: [%.4f] train_acc: [%.4f] test_min_loss: [%.4f] test_acc: [%.4f]'%(t+1, self.train_iters, train_min_loss, train_acc, test_min_loss, test_acc))

				if (t+1)%self.sub_train_iters == 0:
					
					if search_algorithm == 'random_search':						
						print('\n\nRunning Random Search')
						save_file_name=os.path.join(self.images_dir,'Random_string_length_'+str(self.string_length)+'_iter_'+str(t+1))
						min_tr_accuracy, _transformations, _levels, _image = self.search_ops.random_search(100, self.string_length, save_file_name,
																										self.test, source_train_images[:1000],
																										source_train_labels[:1000], sess) 
					
					elif search_algorithm == 'evolution_search':						
						print('\n\nRunning Evolution Search')
						save_file_name=os.path.join(self.images_dir,'Evolution_string_length_'+str(self.string_length)+'_iter_'+str(t+1))
						min_tr_accuracy, _transformations, _levels, _image = self.search_ops.genetic_algorithm(10, 10, self.string_length, 0.1,
																											save_file_name, self.test, 
																											source_train_images[:1000],	source_train_labels[:1000], sess) 
																									
					transformations.append(_transformations)
					levels.append(_levels)

					print('Target accuracy: [%.4f]'%(min_tr_accuracy))
					print('_'.join(_transformations))
					print('\n\n')		

				if (t+1) % 25000 == 0:
					print('Saving')
					saver.save(sess, os.path.join(self.model_save_path, 'encoder'))


	def test(self, images, labels, sess):

		N = 1 #set accordingly to GPU memory
		target_accuracy = 0
		target_loss = 0
		preds = []

		for test_images_batch, test_labels_batch in zip(np.array_split(images, N), np.array_split(labels, N)):
			feed_dict = {self.model.images: test_images_batch, self.model.labels: test_labels_batch} 
			target_accuracy_tmp, target_loss_tmp, pred = sess.run([self.model.accuracy, self.model.min_loss, self.model.pred], feed_dict) 
			target_accuracy += target_accuracy_tmp/float(N)
			target_loss += target_loss_tmp/float(N)
			preds.append(pred.tolist())

		correct_guesses = (np.array(preds)==labels).astype(int)[0]
		
		return target_accuracy, correct_guesses
	
	def test_all(self):

		# build a graph
		print('Building model')
		self.model.build_model()
		print('Built')

		res_dict = dict()
		res_dict['exp_dir'] = self.exp_dir

		print 'Testing ALL'

		targets = ['mnist', 'svhn']# add 'usps', 'syn', 'mnist_m'

		for target in targets:

			print('\n\n\n...........................................................................')

			test_images, test_labels = self.load_test_data(target=target)

			with tf.Session() as sess:

				print('...........................................................................')

				tf.global_variables_initializer().run()

				
				print('Loading pre-trained model.')
				variables_to_restore = slim.get_model_variables()
				restorer = tf.train.Saver(variables_to_restore)
				restorer.restore(sess, os.path.join(self.model_save_path,'encoder'))
				
				N = 100
				target_accuracy = 0
				target_loss = 0

				print('Calculating accuracy')

				for test_images_batch, test_labels_batch in zip(np.array_split(test_images, N), np.array_split(test_labels, N)):
					feed_dict = {self.model.images: test_images_batch, self.model.labels: test_labels_batch} 
					target_accuracy_tmp, target_loss_tmp, target_pred = sess.run([self.model.accuracy, self.model.min_loss, self.model.pred], feed_dict) 
					target_accuracy += target_accuracy_tmp/float(N)
					target_loss += target_loss_tmp/float(N)

			print('Target accuracy: [%.4f] target loss: [%.4f]'%(target_accuracy, target_loss))

			res_dict[target] = target_accuracy

		with open(os.path.join(self.exp_dir, 'domain_generalization_performance.pkl'), 'w') as f:
			cPickle.dump(res_dict, f, cPickle.HIGHEST_PROTOCOL)


	def test_random_search(self, run, seed, no_iters, string_length):

		test_images, test_labels = self.load_test_data(target='mnist')
		
		npr.seed(213)
		rnd_idx = range(len(test_images))
		npr.shuffle(rnd_idx)		

		test_images = test_images[rnd_idx]
		test_labels = test_labels[rnd_idx]

		test_images = test_images[:1000]
		test_labels = test_labels[:1000]

		npr.seed(seed)

		# build a graph
		print 'Building model'
		self.model.mode='train_encoder'
		self.model.build_model()
		print 'Built'

		with tf.Session() as sess:

			tf.global_variables_initializer().run()

			print ('Loading pre-trained model.')
			variables_to_restore = slim.get_model_variables(scope='encoder')
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, os.path.join(self.model_save_path,'encoder'))

			if not os.path.exists(os.path.join(self.exp_dir,'images')):
				os.makedirs(os.path.join(self.exp_dir,'images'))

			# perform random search
			
			save_search_file_name=os.path.join(self.images_dir,'TEST_Random_test_string_length_'+str(string_length))

			all_accuracies, all_transformations, all_levels, all_images = self.search_ops.random_search(5000, string_length, save_search_file_name,	self.test, test_images, test_labels, sess) 
			# save output

			with open(os.path.join(self.exp_dir, 'worst_case_accuracies.pkl'), 'w') as f:
				cPickle.dump((all_accuracies, all_transformations, all_levels), f, cPickle.HIGHEST_PROTOCOL)

	def test_evolution_search(self, run='0', seed=123, no_iters=100, string_length=3, pop_size=10, mutation_rate=0.1):

		test_images, test_labels = self.load_mnist(split='test')
		
		npr.seed(213)
		rnd_idx = range(len(test_images))
		npr.shuffle(rnd_idx)		

		test_images = test_images[rnd_idx]
		test_labels = test_labels[rnd_idx]

		test_images = test_images[:1000]
		test_labels = test_labels[:1000]

		npr.seed(seed)

		# build a graph
		print 'Building model'
		self.model.mode='train_encoder'
		self.model.build_model()
		print 'Built'

		with tf.Session() as sess:

			tf.global_variables_initializer().run()

			print ('Loading pre-trained model.')
			variables_to_restore = slim.get_model_variables()
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, os.path.join(self.model_save_path,'encoder'))

			if not os.path.exists(os.path.join(self.exp_dir,'GA_images')):
				os.makedirs(os.path.join(self.exp_dir,'GA_images'))
						
			save_search_file_name=os.path.join(self.images_dir,'TEST_Evolution_test_string_length_'+str(string_length))

			self.search_ops.genetic_algorithm(100, pop_size, string_length, mutation_rate, save_search_file_name, self.test, test_images, test_labels, sess)

if __name__=='__main__':

    print '...'


