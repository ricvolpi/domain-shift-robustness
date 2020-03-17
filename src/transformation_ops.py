import os
import numpy as np
import numpy.random as npr
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.Image
import matplotlib

class TransfOps(object):

	'''
	Class to handle the decoding of the strings used with the genetic
	algorithm and all the data transformations.
	'''
	
	def __init__(self):
				
		self.transformation_list = ['autocontrast', 'brightness', 'color', 'contrast', 'sharpness', 'solarize', 'grayscale', 'Renhancer', 'Genhancer', 'Benhancer']
		self.define_code_correspondances()
		
	def decode_string(self, transf_string):
		
		'''
		Code to decode the string used by the genetic algorithm
		String example: 't1,l1_3,t4,l4_0,t0,l0_1'. First transformation is the one 
		associated with index '1', with level set to '3', and so on.
		'random_N' with N integer gives N rnd transformations with rnd levels.
		'''
		
		if 'random' in transf_string:
			transformations = npr.choice(self.transformation_list, int(transf_string.split('_')[-1])) # the string is 'random_N'
			levels = [npr.choice(list(self.code_to_level_dict[t].values()), 1)[0] for t in transformations] # list() to make it compatible with Python3
		else:
			transformation_codes = transf_string.split(',')[0::2] 
			level_codes = transf_string.split(',')[1::2]
			
			transformations = [self.code_to_transf(code) for code in transformation_codes] 	
			levels = [self.code_to_level(transf,level) for transf,level in zip(transformations, level_codes)] 	

		return transformations, levels		

	def transform_dataset(self, dataset, transf_string = 't0,l0_0', transformations=None, levels=None):
		
		'''
		dataset: set of images, shape should be N x width x height x #channels
		transf_string: transformations and levels encoded in a string 
		'''

		#print 'Dataset size:',dataset.shape
		
		if len(dataset.shape) == 3: # if 'dataset' is a single image
			dataset = np.expand_dims(dataset, 0) 
		
		if dataset.shape[-1] != 3:
			print('Input shape:', str(dataset.shape))
			raise Exception('The images must be in RGB format')

		tr_dataset = np.zeros((dataset.shape))
		
		if transformations is None:
			# decoding transformation string
			transformations, levels = self.decode_string(transf_string)	
		
		for n,img in enumerate(dataset):
			pil_img = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
			for transf,level in zip(transformations, levels): 
				pil_img = self.apply_transformation(pil_img, transf, level)
			tr_dataset[n] = np.array(pil_img)

		return tr_dataset, transformations, levels

	def apply_transformation(self, image, transformation, level):
		
		'''
		image: image to be tranformed, shape should be 1 x width x height x #channels
		transformation: type of transformation to be applied
		level: level of the perturbation to be applied 
		'''

		if transformation == 'identity':
			return image 

		elif transformation == 'autocontrast':
			return PIL.ImageOps.autocontrast(image, cutoff=level)

		elif transformation == 'brightness':
			return PIL.ImageEnhance.Brightness(image).enhance(level)

		elif transformation == 'color':
			return PIL.ImageEnhance.Color(image).enhance(level)
			
		elif transformation == 'contrast':
			return PIL.ImageEnhance.Contrast(image).enhance(level)

		elif transformation == 'sharpness':
			return PIL.ImageEnhance.Sharpness(image).enhance(level)
			
		elif transformation == 'solarize':
			return PIL.ImageOps.solarize(image, threshold=level)

		elif transformation == 'grayscale':
			image = PIL.ImageOps.grayscale(image).convert('RGB')
			return image		

		elif transformation == 'Renhancer':
			image = np.array(image).astype(int)
			image[:,:,0] += level
			image[image>255] = 255
			image[image<0] = 0
			
			image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
			return image

		elif transformation == 'Genhancer':
			image = np.array(image).astype(int)
			image[:,:,1] += level
			image[image>255] = 255
			image[image<0] = 0
			image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
			return image

		elif transformation == 'Benhancer':
			image = np.array(image).astype(int)
			image[:,:,2] += level
			image[image>255] = 255
			image[image<0] = 0
			image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
			return image

	def code_to_transf(self, code):
		
		'''
		Takes in input a code (e.g., 't0', 't1', ...) and gives in output 
		the related transformation.
		'''

		return self.code_to_transf_dict[code]


	def code_to_level(self, transformation, code):
		
		'''
		Takes in input a transfotmation (e.g., 'invert', 'colorize', ...) and 
		a level code (e.g., 'l0_1', 'l1_3', ...) and gives in output the related level.
		'''
		
		return self.code_to_level_dict[transformation][code]
				
	def define_code_correspondances(self):
		
		'''
		Define the correpondances between transformation/level codes
		and the actual types and values.
		'''
			
		self.code_to_transf_dict = dict()
		
		self.code_to_transf_dict['t1'] = 'autocontrast'
		self.code_to_transf_dict['t2'] = 'brightness'
		self.code_to_transf_dict['t3'] = 'color'
		self.code_to_transf_dict['t4'] = 'contrast'
		self.code_to_transf_dict['t5'] = 'sharpness'
		self.code_to_transf_dict['t6'] = 'solarize'
		self.code_to_transf_dict['t7'] = 'grayscale'
		self.code_to_transf_dict['t8'] = 'Renhancer'
		self.code_to_transf_dict['t9'] = 'Genhancer'
		self.code_to_transf_dict['t10'] = 'Benhancer'

		self.code_to_level_dict = dict()
		
		for k in self.transformation_list:
			self.code_to_level_dict[k] = dict()
			
		# percentages
		self.code_to_level_dict['autocontrast'] = dict()
		for n,l in enumerate(np.linspace(0.0,0.3,20)):
			self.code_to_level_dict['autocontrast']['l1_'+str(n)] = l

		# factors
		self.code_to_level_dict['brightness'] = dict()
		for n,l in enumerate(np.linspace(0.6,1.4,20)):
			self.code_to_level_dict['brightness']['l2_'+str(n)] = l
		
		# factors
		self.code_to_level_dict['color'] = dict()
		for n,l in enumerate(np.linspace(0.6,1.4,20)):
			self.code_to_level_dict['color']['l3_'+str(n)] = l
		
		# factors
		self.code_to_level_dict['contrast'] = dict()
		for n,l in enumerate(np.linspace(0.6,1.4,20)):
			self.code_to_level_dict['contrast']['l4_'+str(n)] = l

		# factors
		self.code_to_level_dict['sharpness'] = dict()
		for n,l in enumerate(np.linspace(0.6,1.4,20)):
			self.code_to_level_dict['sharpness']['l5_'+str(n)] = l
		
		self.code_to_level_dict['solarize'] = dict()
		for n,l in enumerate(np.linspace(0,20,20).astype(int)):
			self.code_to_level_dict['solarize']['l6_'+str(n)] = l

		self.code_to_level_dict['grayscale']['l7_0'] = None

		# percentages
		self.code_to_level_dict['Renhancer'] = dict()
		for n,l in enumerate(np.linspace(-120,120,30).astype(int)):
			self.code_to_level_dict['Renhancer']['l8_'+str(n)] = l

		# percentages
		self.code_to_level_dict['Genhancer'] = dict()
		for n,l in enumerate(np.linspace(-120,120,30).astype(int)):
			self.code_to_level_dict['Genhancer']['l9_'+str(n)] = l

		# percentages
		self.code_to_level_dict['Benhancer'] = dict()
		for n,l in enumerate(np.linspace(-120,120,30).astype(int)):
			self.code_to_level_dict['Benhancer']['l10_'+str(n)] = l

if __name__=='__main__':
	pass








