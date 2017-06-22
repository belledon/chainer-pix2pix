import sys
import numpy as np
from random import sample 
from math import factorial, ceil 
from itertools import combinations, repeat
from time import clock

import h5py
from PIL import Image
from chainer.dataset import DatasetMixin

from scipy.ndimage import zoom 


import fileFuncs as ff
from downsample import loadBinvox, rotate_matrix, quat


#-----------------------------------------------------------------#
#                         	  Helpers                             #
#-----------------------------------------------------------------#

def reject_sample(s, k, n):
	# n combinations of size k
	ctr = 0
	results = []


	while ctr < n:
		inds = np.random.choice(s, k, replace=False)

		if ctr == 0 or all(np.sum(np.abs(np.array(results) - inds), 
			axis=1) !=0):

			results.append(inds)
			ctr += 1

	return np.array(results)


def repeat_samples(array, n):
	# N is the ratio of repeats
	assert(n > 1.0)
	n = ceil(n)
	repeats = np.repeat(array, n, axis=0)
	np.random.shuffle(repeats)
	return repeats

def rand_pad(array, n, pad = 0.0):
	array = np.array(array)
	l = len(array)
	m = n - l 
	assert(m >= 0)
	if m == 0:
		return l
	else:
		pad = (m , array.shape[1])
		t = np.zeros(pad)
		t = list(array) + list(t)
		return np.array(sample(t, n))

#-----------------------------------------------------------------#
#                        Dataset Class                            #
#-----------------------------------------------------------------#

class CrossModalDataset(DatasetMixin):
	# Dataset Structure:
	# - each object is a group within hdf5
	# 
	# 
	# 
	in_types = ["images", "grasps", "multisensory"]
	out_types = ["id", "voxels",  "all"]
	grasp_max = 8

	def __init__(self, path, in_type, out_type, 
		rot = False,  grasp_count = 6,
		in_size = 10, image_size = (256, 256), synset = None,
		debug = False, test = False):

		assert(in_type in CrossModalDataset.in_types and
			out_type in CrossModalDataset.out_types)

		self.debug = debug
		self.in_type = in_type
		self.out_type = out_type
		self.in_size = in_size
		self.rot = rot if in_type in ["images", "mutlisensory"] else False
		self.grasp_count = grasp_count
		self.image_size = image_size
		self.synset = synset
		self.test = test
		self._initialize(path)
		
	def __len__(self):
		return self.size

	# int -> sample
	def get_example(self, index):

		in_data = self.in_data_func(index)
		out_data = self.out_data_func(index)

		if isinstance(in_data, tuple) and not isinstance(out_data , tuple):
			return in_data +(out_data,)

		else:
			return in_data, out_data	

	#-----------------------------------------------------------------#
	#                         Initialization                          #
	#-----------------------------------------------------------------#

	def _initialize(self, fp):
		
		norm_dir = ff.sameDir(fp, "norms")
		print("Searching for norms in {}".format(norm_dir))
		
		f_dir = lambda x: ff.join(norm_dir, x)
		grasp_norms = f_dir("norms.npy")
		image_mean = f_dir("image_mean.npy")

 
		assert(ff.isFile(grasp_norms))
		self.grasp_mean = np.load(grasp_norms)

		if ff.isFile(image_mean):
			self.means = True

			image_mean = np.load(image_mean).squeeze()
			image_mean = Image.fromarray(image_mean).resize((self.image_size))
			self.image_mean = np.asarray(image_mean)
		
		else:
			self.means = False
			self.image_mean = 0.0
			
			print("NOT using data normalization. Not"+\
				" recommended to train on this data")

		if self.synset is not None:
			fileList, ind = self.synset
			fileList= np.load(fileList)
			splits = np.array([f.split("/") for f in fileList])
			self.synsetList = splits[:, -3]
			self.synset = np.unique(self.synsetList)[ind]
			print("Selecting for synset: {}".format(self.synset))		

		print("Loading dataset...")
		self._initialize_data(fp)
		print("Assigning retrievers...")
		self._initialize_get_funcs()
		print("Initialization complete")
		print("Created dataset with {} examples".format(self.size))

	def _initialize_data(self, fp):

		ctr = 0
		data = {}
		i = 0


		with h5py.File(fp, 'r') as f:

			
			for obj in f:
				
				f_data = f[obj]
				obj_data = {}

				obj_data[str(i)] = obj
				# Optionally exclude non-synset objects
				if self.synset is not None:
					ind = int(f_data["id"].value)
					if self.synsetList[ind] != self.synset:
						continue

				if self.in_type == "multisensory":

					im_data, im_ctr = self._init_data(f_data, "images")
					
					grasp_data, grasp_ctr = self._init_data(f_data, "grasps")
					assert(im_ctr == grasp_ctr)
					size = im_ctr
					obj_data.update(im_data)
					obj_data.update(grasp_data)
					
					
				else:
					t, size = self._init_data(f_data, self.in_type)
					obj_data.update(t)

				# Each object must have the same
				# number of samples
				ctr += size
				try:
					assert(ctr % size == 0)
				except:
					msg = ("Object {} did not have the proper " +\
					"number of {} ({}/{})").format(obj, self.in_type, 
					size, CrossModalDataset.in_size)
					raise ValueError(msg)

				if self.out_type == "all":
					# Copy over the label data
					out_data = f_data["id"].value
					obj_data["id"] = out_data

					out_data = f_data["voxels"].value
					obj_data["voxels"] = out_data
				else:
					out_data = f_data[self.out_type].value
					obj_data[self.out_type] = out_data

				
				# Rotation data if stated
				if self.rot:
					rot_data = f[obj]["rotations"][:]
					obj_data["rotations"] = rot_data

				data[str(i)] = obj_data
				i += 1

				if self.debug:
					print("DEBUG MODE IS ON! ONLY USING FIRST OBJECT")
					break

		self.size = ctr
		self.data = data
		self.num_objs = len(data.keys())

	def _init_data(self, dataset, t):

		obj_data = {}

		in_data = np.squeeze(dataset[t])

		# Return the indeces for grasps
		if t == "grasps":

			in_data = reject_sample(54, self.grasp_count, self.in_size)
			
		size = len(in_data)
		
		obj_data[t] = in_data
		
		return obj_data, size



	def _initialize_get_funcs(self):

		if self.in_type == "images":
			self.in_data_func = self.get_image

		elif self.in_type == "grasps":
      
			self.in_data_func = self.get_grasp
      
		else:
    # multisensory
			self.in_data_func = lambda x: (self.get_image(x), self.get_grasp(x))

		if self.out_type == "id":
			self.out_data_func = self.get_id
		
		elif self.out_type == "all":
			self.out_data_func = lambda x: (self.get_id(x), self.get_voxel(x))

		else:
		# voxels
			self.out_data_func = self.get_voxel

		# if self.rot:
		# 	t = self.out_data_func
		# 	self.out_data_func = lambda x: (t(x), self.get_rot(x))

	#-----------------------------------------------------------------#
	#                         General Helpers                         #
	#-----------------------------------------------------------------#
	
	def get_ind(self, i):
		obj = str(int(i / self.in_size))
		ind = i % self.in_size
		return obj, ind

	def get_data(self, i, type):
		obj, ind = self.get_ind(i)
		try:
			data = self.data[obj][type][ind]
			
		except:
			msg = "Tried to get data from obj {} of type {} at {}".format(
				obj, type, ind)
			raise IndexError(msg)

		return data
	#-----------------------------------------------------------------#
	#                           Data specific                         #
	#-----------------------------------------------------------------#

	# There may be a more beautiful way than writing "get_data"
	# everywhere but it isn't apperant to me write now.

	def get_image(self, i):
		# t0 = clock()
		file = self.get_data(i, "images")
		with Image.open(file).resize(self.image_size) as f:
			image = np.asarray(f, dtype=np.float32)
		
		image = image[:,:,0:3]
		image = np.mean(image, axis = 2)

		if not self.test:

			image *= (1.0 / 255.0)
			image -= self.image_mean
		
		
		image = np.expand_dims(image, axis = 0)

		# print(image.shape)
		# print("get_image took {}s".format(clock()-t0))
		return image
	

		
	def get_rot(self, i):
		rots = self.get_data(i, "rotations")
		rots = quat(*rots)
		# rots -= self.rot_mean
		# rots *= 1 / np.pi
		return rots.astype(np.float32)

	def get_grasp(self, i):
		# t0 = clock()
		obj, _ = self.get_ind(i)
		obj = int(obj)
		inds = self.get_data(i, "grasps").astype(np.int)
		grasps = self.grasp_mean[obj][inds]
		grasps = np.expand_dims(grasps, axis=0)
		# print("get_grasp took {}s".format(clock()-t0))
		return grasps.astype(np.float32)

	def get_id(self, i):
		obj, _ = self.get_ind(i)
		# obj = np.array(int(self.data[obj]["id"]/50) ).astype(np.int32)
		obj = np.array(obj).astype(np.int32)
		return obj

	def get_voxel(self, i):
		# t0 = clock()
		obj, _ = self.get_ind(i)
		voxels = self.data[obj]["voxels"]

		if isinstance(voxels, bytes):
			voxels = loadBinvox(voxels).data
		
		# if self.rot:
		# 	rot = self.get_rot(i)
		# 	voxels = rotate_matrix(voxels, rot)

		# for tanh
		voxels = voxels.astype(np.float32)
		voxels = np.expand_dims(voxels, axis = 0)
		# return voxels.astype(np.int32)
		return voxels

