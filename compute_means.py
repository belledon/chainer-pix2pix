import argparse
import sys
import h5py
import numpy as np

from data import CrossModalDataset

import fileFuncs as ff

def compute_mean(dataset):
    
    in_sum = 0
   
    n = len(dataset)
    
    for index in range(n):
        msg = "On {}/{} for {}".format(index, n, dataset.in_type)
        print("", end = '\r')
        print(msg, end = '\r')
        data, label = dataset.get_example(index)
        in_sum += data 


    return in_sum / n


def main():
    parser = argparse.ArgumentParser(description='Compute images mean array')
    parser.add_argument('dataset', type = str,
                        help='Path to training image-label list file')
    
    parser.add_argument("in_size", type = int, 
                        help = "Expected number of samples per object")
    args = parser.parse_args()
    
    # images + rotations
    dataset = CrossModalDataset(args.dataset, "images", "id",
        in_size = args.in_size)
    image_mean = compute_mean(dataset)
    np.save(ff.sameDir(args.dataset, ff.join("norms", "image_mean.npy")), image_mean)


    print("done")
    



if __name__ == '__main__':
    main()