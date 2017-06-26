import argparse

import fileFuncs as ff

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

def cuboid_data(pos, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return x, y, z

def plotCubeAt(pos=(0,0,0),ax=None):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos )
        ax.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=1)

def plotMatrix(ax, matrix):
    # plot a Matrix 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i,j,k] == 1:
                    # to have the 
                    plotCubeAt(pos=(i-0.5,j-0.5,k-0.5), ax=ax)            



def save_voxels(f, voxels):
	fig = plt.figure(frameon=False)
	ax = fig.gca(projection='3d')
	ax.set_aspect('equal')

	plotMatrix(ax, voxels)
	ax.set_xlim([0,64])
	ax.set_ylim([0,64])
	ax.set_zlim([0,64])
	fig.savefig(f)
	fig.clear()
	return 

def save_image(f, array):
	print(array.shape)
	image = Image.fromarray(array, mode="F").convert("RGB")
	image.save(f)

def main():

	parser = argparse.ArgumentParser(
		description= "Visualized voxel maps")

	parser.add_argument("voxels", type = str, 
		help = "Path to voxel npys")

	parser.add_argument('iter', type = int, 
        help='Iteration number for network')

	parser.add_argument("out", type = str,
		help = "Path to save voxels")

	parser.add_argument("--check_dataset", "-cd", type = int, default = 0,
        help = "Returns input, ground truth used for testing")

	args = parser.parse_args()

	ff.ensureDir(args.out)

	voxels = np.load(args.voxels)

	for i, v_map in enumerate(voxels):

		print("Object {}/{}".format(i+1, len(voxels)))
		rout = ff.join(args.out, "recon_{}.png".format(i))
		save_voxels(rout, v_map)

		if args.check_dataset:

			t_in = ff.join(ff.fileDir(args.voxels), 
				"check_trial/{}_{}_{}.npy".format(args.iter, i,"t"))
			x_in = ff.join(ff.fileDir(args.voxels), 
				"check_trial/{}_{}_{}.npy".format(args.iter, i,"x"))

			t = np.load(t_in).squeeze()
			x = np.load(x_in).squeeze()

			tout = ff.join(args.out, "gt_{}.png".format(i))

			if t.ndim == 3:
				save_voxels(tout, t)
				
			else:
				t_gt = t[0]
				t_d = t[1]
				save_voxels(tout, t_gt)

				d_out = ff.join(args.out, "d_{}.png".format(i))
				save_voxels(d_out, t_d)
			
		
			iout = ff.join(args.out, "x_{}.png".format(i))
			save_image(iout, x)



if __name__ == '__main__':
	main()