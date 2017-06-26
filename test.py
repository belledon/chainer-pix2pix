#!/usr/bin/env python


from __future__ import print_function
import argparse
import os
import sys
import numpy as np 

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

import fileFuncs as ff
import data
import net


archs = {   "voxel-voxel" : net.voxeltovoxel,
            "pix-voxel" : net.pixtovoxel,
        }

def iou(a, b):
   
    summed = a + b
    intersect = np.sum(summed == 2)
    union = np.sum(summed > 0)
    return intersect /union

def predict(obs, thresh):

    
    xp = chainer.cuda.get_array_module(obs)
    result = obs.data
    t = np.zeros(result.shape)
    t[np.where(result >= thresh)] = 1
    print("RECON MEAN: {}".format(result.mean()))
    print("Recon contains {} voxels".format(t.sum()))
    return t

def probe(model, data, thresh):

    enc, dec  = model 

    x,t = data 
    print("T MEAN: {}".format(t.mean()))
    xp = enc.xp
    x = chainer.Variable(xp.expand_dims(x, axis=0))
    # with chainer.using_config('train', False):
    #     recon = dec(enc(x))
    #     prediction = predict(recon, thresh)
    recon = dec(enc(x))
    prediction = predict(recon, thresh)

    print(iou(prediction, t))
    return prediction



def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')


    parser.add_argument("database", type = str, 
        help = "Path for testing database")

    parser.add_argument("arch", type = str, choices = archs.keys(),
        help = "Network Architecture")

    parser.add_argument('net', type = str, 
        help='Path to trained network')

    parser.add_argument('iter', type = int, 
        help='Iteration number for network')

    parser.add_argument("--check_dataset", "-cd", type = int, default = 1,
        help = "Returns input, ground truth used for testing")

    parser.add_argument("--rot", "-r", type = int, default = 0,
        help = "Include rotations for images -> *")

    parser.add_argument("--voxels", "-v", type = int, default = 64,
        help = "Number of voxels (default 64)")

    parser.add_argument("--image", "-is", type = int, default = 256,
        help = "Size of images. Default(256)")

    parser.add_argument('--out', '-o', type = str, default=os.path.join(os.getcwd(), "nn_logs"),
        help='Directory to output the result')

    parser.add_argument('--gpu', '-g', type=int, default=-1,
        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--example_count', '-ec', type=int, default=10,
        help='Number of samples per object')

    parser.add_argument('--synset_list', '-syn', type = str,
        help = "Path to synset list")

    parser.add_argument('--synset_id', '-sid', type = int,
        help = "Index of synset")

    parser.add_argument("--number", "-n", type = int, 
        help = "Number of objects to test")
    args = parser.parse_args()

    print("################################")
    print("Load network from: {}".format(args.out))
    print("##### Data #####")
    print("Using database: {}".format(args.database))
    print("Voxels : {0:d}x{0:d}x{0:d}".format(args.voxels))
    print("Image size : 1x{0:d}x{0:d}".format(args.image))
    if args.check_dataset:
        print("Checking dataset")
    print("################################")
    sys.stdout.flush()

    # Set up a neural network to test


    arch = archs[args.arch]
    enc = arch.Encoder()
    dec = arch.Decoder()
    # dis = arch.Discriminator()

    

    f_name = lambda m: ff.join(args.net, "{}_iter_{}.npz".format(m,args.iter))
    serializers.npz.load_npz(f_name("enc"), enc)
    serializers.npz.load_npz(f_name("dec"), dec)

    model = (enc, dec)
    # load_npz(f_name("dis"), dis)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()


    if args.synset_list is not None:
        assert(args.synset_id is not None)
        syn = (args.synset_list, args.synset_id)
    else:
        syn = None

    dataset = data.CrossModalDataset(
        args.database, "images", "voxels", in_size=args.example_count, rot=args.rot, 
        image_size = (args.image, args.image), synset = syn, test = False
    )    
        
    

    probe_f = lambda d: probe(model, d, 0.5)

    if args.number is not None:
        n = args.number
    else:
        n = len(dataset)

    recons = np.zeros((n, args.voxels, args.voxels, args.voxels))

    for i in range(n):
        print("Object {}/{}".format(i+1, n))

        xt = dataset.get_example(i)

        if args.check_dataset:
            x,t = xt
            ff.ensureDir(ff.join(args.out, "check_trial"))
            xout = ff.join(args.out, ff.join("check_trial","{}_{}_{}.npy".format(args.iter, i, "x")))
            tout = ff.join(args.out, ff.join("check_trial","{}_{}_{}.npy".format(args.iter, i, "t")))
            np.save(xout, x)
            np.save(tout, t)

        recons[i] = probe_f(xt)

    out = ff.join(args.out, "reconstructions_{}.npy".format(args.iter))
    np.save(out, recons)




if __name__ == '__main__':
    main()
