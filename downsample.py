import argparse
import numpy as np
from transforms3d import quaternions
import fileFuncs as ff

import importlib
binvox_rw = importlib.import_module("binvox-rw-py.binvox_rw")

# ---------------------------------------------------------------------------------------------
def cubify(arr, m):
    
    n = int(arr.shape[0]/m)
    order = np.arange(n)
    temp = np.zeros(np.repeat(n, 3))

    for i in order:
    	x = i*m
    	for j in order:
    		y = j*m
    		for k in order:
    			z = k*m
    			cube = arr[x:x+m, y:y+m, z:z+m]
    			avg = np.mean(cube)
    			temp[i,j,k] = avg > 0.0
    return temp


# ---------------------------------------------------------------------------------------------

def downsample(data, factor):
	assert(data.ndim == 3)
	assert(sum(data.shape)/3 == data.shape[0])
	assert(data.shape[0] % factor == 0)
	assert(data.shape[0] / factor > 0)

	cubes = cubify(data, factor).astype(np.bool)
	
	return cubes

def build_voxels(data, voxels):
	return binvox_rw.Voxels(data, data.shape, voxels.translate, 
		voxels.scale, voxels.axis_order)

def loadBinvox(file):
	with open(file, "rb") as f:
		voxels = binvox_rw.read_as_3d_array(f)
	return voxels

def rotate_matrix(matrix, q):

	rot = quaternions.quat2mat(q)
	# convert booleans to xyz coordinates
	inds = np.vstack(np.nonzero(matrix)).T
	# Rotate xyz coordinates
	d = inds.dot(rot).astype(np.int)
	# Shift by the rotation
	neg_shift = np.amin(d, axis = 0)
	d -= neg_shift
	d = np.array([r for r in d if all((matrix.shape - r) > 0)])
	# Lift indeces to boolean matrix
	rotated = np.zeros(matrix.shape)
	rotated[[d[:, 0], d[:, 1], d[:, 2]]] = 1
	data = rotated > 0

	return data

def quat(x,y,z):
	# create a quaternion for each rotation along each axis
	xr = quaternions.axangle2quat([1,0,0], x)
	yr = quaternions.axangle2quat([0,1,0], y)
	zr = quaternions.axangle2quat([0,0,1], z)
	return quaternions.qmult(quaternions.qmult(xr,yr), zr)

def main():

	parser = argparse.ArgumentParser(
		description = "Downsamples binvox file")

	parser.add_argument("path", type = str,
		help = "Path to binvox file")

	parser.add_argument("out", type = str,
		help = "Path to save downsampled binvox")
	
	parser.add_argument("--factor", '-f', type = int, default = 4,
		help = "Factor to downsample by")

	parser.add_argument("--rot", "-r", type = float, nargs="+",
		help = "Rotation applied")

	parser.add_argument("--start", "-st", type = int,
		help = "If reading from file_list, start index")

	parser.add_argument("--stop", "-sp", type = int,
		help = "If reading from file_list, stop index")
	
	args = parser.parse_args()

	assert(ff.isFile(args.path))
	ff.ensureDir(args.out) 

	if ff.fileExt(args.path) == ".npy":
		assert(isinstance(args.start, int) and isinstance(args.stop, int))
		paths = np.load(args.path)[args.start : args.stop]
		print("Succesfully loaded file_list. Rendering:")
		print(paths)
	else:
		paths = [args.path]

	for path in paths:

		out = ff.swapDir(path, args.out)

		# if ff.isFile(out):
		# 	print("File {} already exists. Skipping".format(
		# 		out))
		# 	continue

		with open(path, "rb") as f:
			voxels = binvox_rw.read_as_3d_array(f)

		downsampled = downsample(voxels.data, args.factor)

		if args.rot is not None:
			downsampled = rotate_matrix(downsampled, quat(*args.rot))

		newVoxels = build_voxels(downsampled, voxels)

		

		with open(out, "wb") as f:
			newVoxels.write(f)

	print("done")

if __name__ == '__main__':
	main()