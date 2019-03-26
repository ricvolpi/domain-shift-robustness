import tensorflow as tf
from model import Model
from train_ops import TrainOps
import glob
import os
import cPickle

import numpy.random as npr
import numpy as np
import itertools

flags = tf.app.flags
flags.DEFINE_string('gpu', '0', "GPU to used")
flags.DEFINE_string('exp_dir', 'exp_dir', "Experiment directory")
flags.DEFINE_string('mode', 'mode', "Experiment directory")
flags.DEFINE_string('run', '0', "Experiment run")
flags.DEFINE_string('seed', '123', "Random seed")

flags.DEFINE_string('transf_string_length', '5', "Number of transformations to be concatenated")

flags.DEFINE_string('search_no_iters', '100', "Number of search iterations")
flags.DEFINE_string('GA_pop_size', '10', "Number of individuals in the population -- for evolution search")
flags.DEFINE_string('GA_mutation_rate', '0.1', "Mutation rate -- for evolution search")
FLAGS = flags.FLAGS

def main(_):

	#npr.seed(int(FLAGS.seed))

	GPU_ID = FLAGS.gpu
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 on stackoverflow
	os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

	EXP_DIR = FLAGS.exp_dir

	model = Model()
	tr_ops = TrainOps(model, EXP_DIR)

	if 'train' in FLAGS.mode:
		npr.seed(int(FLAGS.seed))

	
	if FLAGS.mode=='train_ERM':
		print 'Training model with standard ERM'
		tr_ops.load_exp_config()
		tr_ops.train()      
		
	if FLAGS.mode=='train_RDA':
		print 'Training model with RDA'
		tr_ops.load_exp_config()
		tr_ops.train(random_transf=True)
		
	if FLAGS.mode=='train_RSDA':
		print 'Training model with RSDA'
		tr_ops.load_exp_config()
		tr_ops.train_search(search_algorithm='random_search')
		
	if FLAGS.mode=='train_ESDA':
		print 'Training model with ESDA'
		tr_ops.load_exp_config()
		tr_ops.train_search(search_algorithm='evolution_search')

		
	elif FLAGS.mode=='test_all':
		print 'Testing all'
		tr_ops.load_exp_config()
		tr_ops.test_all()

	elif FLAGS.mode=='test_RS':
		print 'Random search'
		tr_ops.load_exp_config()
		tr_ops.test_random_search(run=str(FLAGS.run), seed=int(FLAGS.seed), no_iters=int(FLAGS.search_no_iters), string_length=int(FLAGS.transf_string_length)) 

	elif FLAGS.mode=='test_ES':
		print 'Evolution search'
		tr_ops.load_exp_config()
		tr_ops.test_evolution_search(run=str(FLAGS.run), seed=int(FLAGS.seed), no_iters=int(FLAGS.search_no_iters),string_length=int(FLAGS.transf_string_length), 
										pop_size=int(FLAGS.GA_pop_size), mutation_rate=float(FLAGS.GA_mutation_rate))

if __name__ == '__main__':
	tf.app.run()

