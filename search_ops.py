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

from transformation_ops import TransfOps

class SearchOps(object):
	
	'''
	Class to handle all the search procedures.
	Currently implemented: random search and evolution search
	'''

	def __init__(self):
		self.transf_ops = TransfOps()

	def random_search(self, no_iters, string_length, save_file_name, compute_fitness_f, original_images, *args):

		'''
		Sampling random image transformations and testing them on a provided model.
		Referring to the paper, this is Algorithm 1.
		
			no_iters: number of iterations.
			string_length: number of transformations to be concatenated.
			save_file_name: file name used to save .png and .pkl outputs.
			compute_fitness_f: test function associated with the desired model.
			original_images: images to give in input to compute_fitness_f.
			args: other input eventually required by compute_fitness_f (e.g., ground truth labels, sess, etc.)

		'''

		all_accuracies = []
		all_best_accuracies = []
		all_transformations = []
		all_levels = []
		all_images = []
		current_minimum = 1.
		
		number_fitness_evals = 0		
		
		for t in range(no_iters):

			if (t%100)==0:
				print('Iter #',str(t))

			tr_images, transformations, levels = self.transf_ops.transform_dataset(original_images * 255., transf_string='random_'+str(string_length))
			
			tr_images /= 255.
			
			N = 1 #set accordingly to GPU memory
			target_accuracy = 0
			target_loss = 0

			number_fitness_evals += 1

			(target_accuracy, preds) = compute_fitness_f(tr_images, *args)

			all_accuracies.append(target_accuracy)

			if target_accuracy < current_minimum:
				print ('%d Current minimum: [%.4f], # fitness evals: [%d]'%(t, target_accuracy, number_fitness_evals))
				current_minimum=target_accuracy				
			
				all_best_accuracies.append(target_accuracy)
				all_transformations.append(transformations)
				all_levels.append(levels)

				for n, pred in enumerate(preds):
					tr_images[n][:5,:5,:] = 0.
					tr_images[n][:5,:5,1] = pred
					tr_images[n][:5,:5,0] = (1.-pred)
								
				conc_images=np.vstack((np.hstack((tr_images[i]*255. for i in range(j*20, (j+1)*20))) for j in range(10)))
				all_images.append(conc_images)

				PIL.Image.fromarray(conc_images.astype('uint8')).save(save_file_name+'.png')
				
				with open(save_file_name+'.pkl', 'wb') as f:
					cPickle.dump((all_accuracies, all_best_accuracies, all_transformations, all_levels, number_fitness_evals), f, cPickle.HIGHEST_PROTOCOL)

			if (t%100) == 0:
				with open(save_file_name+'.pkl', 'wb') as f:
					cPickle.dump((all_accuracies, all_best_accuracies, all_transformations, all_levels, number_fitness_evals), f, cPickle.HIGHEST_PROTOCOL)


		with open(save_file_name+'.pkl', 'wb') as f:
			cPickle.dump((all_accuracies, all_best_accuracies, all_transformations, all_levels, number_fitness_evals), f, cPickle.HIGHEST_PROTOCOL)
		
		return all_best_accuracies[-1], all_transformations[-1].tolist(), all_levels[-1], all_images[-1]
			
	def genetic_algorithm(self, no_iters, pop_size, string_length, mutation_rate, save_file_name, compute_fitness_f, original_images, *args):

		'''
		Sampling random image transformations and testing them on a provided model.
		Referring to the paper, this is Algorithm 2.
		
			no_iters: number of iterations.
			string_length: number of transformations to be concatenated.
			mutation_rate: a value in [0.0,1.0]
			save_file_name: file name used to save .png and .pkl outputs.
			compute_fitness_f: test function associated with the desired model.
			original_images: images to give in input to compute_fitness_f.
			args: other input eventually required by compute_fitness_f (e.g., ground truth labels, sess, etc.)

		'''

		min_accuracy = 1.0 # initialized with the maximum value
		current_minimum = 1.0 # initialized with the maximum value

		number_fitness_evals = 0
		number_fitness_needed = pop_size

		pop_accuracies = []
		pop_probabilities = []
		pop_transformations = []
		pop_levels = []
		pop_images = []

		min_accs = []
		min_transfs = []
		min_levels = []
		min_images = []

		all_fitnesses = []

		print 'Initializing population'
			
		for p in range(pop_size): # number of items in the population

			tr_images, transformations, levels = self.transf_ops.transform_dataset(original_images * 255., transf_string='random_'+str(string_length))
			tr_images /= 255.

			N = 1 #set accordingly to GPU memory
			target_accuracy = 0
			target_loss = 0

			number_fitness_evals += 1
			
			(target_accuracy, preds) = compute_fitness_f(tr_images, *args)

			pop_accuracies.append(target_accuracy)
			pop_transformations.append(transformations)
			pop_levels.append(levels)
			
			for n, pred in enumerate(preds):
				tr_images[n][:5,:5,:] = 0.
				tr_images[n][:5,:5,1] = pred
				tr_images[n][:5,:5,0] = (1.-pred)
			
			conc_images=np.vstack((np.hstack((tr_images[i]*255. for i in range(j*20, (j+1)*20))) for j in range(10)))
			pop_images.append(conc_images)
			
		pop_probabilities = (1. - np.array(pop_accuracies))/np.sum(1. - np.array(pop_accuracies)) 

		current_minimum = np.min(pop_accuracies)
		print 'Current minimum:',str(current_minimum), '# fitness evals', str(number_fitness_evals)

		min_accs.append(current_minimum)

		all_fitnesses.append(current_minimum)
		
		pop_transformations = [arr.tolist() for arr in pop_transformations]			

		min_transfs.append(pop_transformations[np.argmin(pop_accuracies)])
		min_levels.append(pop_levels[np.argmin(pop_accuracies)])
		min_images.append(pop_images[np.argmin(pop_accuracies)])


		with open(save_file_name+'.pkl', 'wb') as f:
			cPickle.dump((min_accs,min_transfs, min_levels, number_fitness_evals, all_fitnesses), f, cPickle.HIGHEST_PROTOCOL)

		PIL.Image.fromarray(np.squeeze(min_images[0]).astype('uint8')).save(save_file_name+'.png')
	
		print 'Running evolution search'

		for step in range(no_iters): # number of iters for the evolution search

			if current_minimum == 0.0:
				break

			new_pop_accuracies = []
			new_pop_images = []
			new_pop_transformations = [None for i in range(pop_size)]
			new_pop_levels = [None for i in range(pop_size)]

			for p in range(pop_size/2):
					
				# randomly choose two parents to be mated <3
					
				idx_1 = npr.choice(pop_size, p=pop_probabilities)
				idx_2 = npr.choice(pop_size, p=pop_probabilities)

				transformations_1 = pop_transformations[idx_1]
				transformations_2 = pop_transformations[idx_2]
				levels_1 = pop_levels[idx_1]
				levels_2 = pop_levels[idx_2]
				
				# cutting transformations/levels on a random point and 
					
				crossover_point = npr.randint(string_length)
					
				new_transformations_1 = transformations_1[:crossover_point] + transformations_2[crossover_point:]
				new_levels_1 = levels_1[:crossover_point] + levels_2[crossover_point:]
				new_transformations_2 = transformations_2[:crossover_point] + transformations_1[crossover_point:]
				new_levels_2 = levels_2[:crossover_point] + levels_1[crossover_point:]
				
				# adding the new offspring to the new population				

				new_pop_transformations[p] = new_transformations_1
				new_pop_levels[p] = new_levels_1
				new_pop_transformations[p+pop_size/2] = new_transformations_2
				new_pop_levels[p+pop_size/2] = new_levels_2
							
			# mutating some genes
				
			for i, transformations in enumerate(new_pop_transformations):
				for j, transf in enumerate(transformations):
					if npr.rand() < mutation_rate: 
						new_pop_transformations[i][j] = npr.choice(self.transf_ops.transformation_list, 1)[0]
						new_pop_levels[i][j] = npr.choice(self.transf_ops.code_to_level_dict[new_pop_transformations[i][j]].values(), 1)[0]
																
			# computing accuracies ("fitness" values)

			for transformations, levels in zip(new_pop_transformations, new_pop_levels): 

				tr_images, _, _ = self.transf_ops.transform_dataset(original_images * 255., transformations=transformations, levels=levels)
				tr_images /= 255.
				
				N = 1 #set accordingly to GPU memory
				target_accuracy = 0
				target_loss = 0
			
				number_fitness_evals += 1
			
				(target_accuracy, preds) = compute_fitness_f(tr_images, *args)
				
				new_pop_accuracies.append(target_accuracy)


				for n, pred in enumerate(preds):
					tr_images[n][:5,:5,:] = 0.
					tr_images[n][:5,:5,1] = pred
					tr_images[n][:5,:5,0] = (1.-pred)

				conc_images=np.vstack((np.hstack((tr_images[i]*255. for i in range(j*20, (j+1)*20))) for j in range(10)))
				new_pop_images.append(conc_images)
			
			
			pop_transformations = new_pop_transformations
			pop_levels = new_pop_levels
			pop_accuracies = new_pop_accuracies


			pop_images = new_pop_images
			pop_probabilities = (1. - np.array(pop_accuracies))/np.sum(1. - np.array(pop_accuracies)) 

			if np.min(pop_accuracies) < current_minimum:
				current_minimum = np.min(pop_accuracies)
				print str(step), '- Current minimum:', str(current_minimum), '#number fitness evals', str(number_fitness_evals) 
				print pop_transformations[np.argmin(pop_accuracies)]
				print pop_levels[np.argmin(pop_accuracies)]

				number_fitness_needed = number_fitness_evals

				min_accs.append(current_minimum)
				min_transfs.append(pop_transformations[np.argmin(pop_accuracies)])
				min_levels.append(pop_levels[np.argmin(pop_accuracies)])
				min_images.append(pop_images[np.argmin(pop_accuracies)])

				PIL.Image.fromarray(pop_images[np.argmin(pop_accuracies)].astype('uint8')).save(save_file_name+'.png')
				
				with open(save_file_name+'.pkl', 'wb') as f:
					cPickle.dump((min_accs,min_transfs, min_levels, number_fitness_needed, all_fitnesses), f, cPickle.HIGHEST_PROTOCOL)
						
			all_fitnesses.append(current_minimum)

		with open(save_file_name+'.pkl', 'wb') as f:
			cPickle.dump((min_accs,min_transfs, min_levels, number_fitness_needed, all_fitnesses), f, cPickle.HIGHEST_PROTOCOL)

		return min_accs[np.argmin(min_accs)], min_transfs[np.argmin(min_accs)], min_levels[np.argmin(min_accs)], min_images[np.argmin(min_accs)]

if __name__=='__main__':

    print '...'


