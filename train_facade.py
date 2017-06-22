#!/usr/bin/env python


from __future__ import print_function
import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers


from updater import FacadeUpdater

import data
import net

archs = {   "pix-pix" : net.pixtopix,
            "pix-voxel" : net.pixtovoxel,}


def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')


    parser.add_argument("database", type = str, 
        help = "Path for database")

    parser.add_argument("arch", type = str, choices = archs.keys(),
        help = "Network Architecture")

    parser.add_argument("--rot", "-r", type = bool, default = False,
        help = "Include rotations for images -> *")

    parser.add_argument("--voxels", "-v", type = int, default = 64,
        help = "Number of voxels (default 64)")

    parser.add_argument("--image", "-is", type = int, default = 256,
        help = "Size of images. Default(256)")

    parser.add_argument('--batchsize', '-b', type=int, default=10,
        help='Number of samples in each mini-batch')

    parser.add_argument('--epoch', '-e', type=int, default=50,
        help='Number of sweeps over the dataset to train')

    parser.add_argument('--ratio', '-p', type=float, default=0.90,
        help='Ratio for splitting training and test data (Default: 0.90)')

    parser.add_argument('--seed', type=int, default=0,
        help='Random seed')

    parser.add_argument('--snapshot_interval', type=int, default=10,
        help='Interval of snapshot')

    parser.add_argument('--display_interval', type=int, default=1,
        help='Interval of displaying log to console')

    parser.add_argument('--out', '-o', type = str, default=os.path.join(os.getcwd(), "nn_logs"),
        help='Directory to output the result')

    parser.add_argument("--resume", "-re", type = str,
        help = "Path to load network and resume training")

    parser.add_argument('--gpu', '-g', type=int, default=-1,
        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--loaderjob', '-j', type=int, default=2,
        help='Number of parallel data loading processes')

    parser.add_argument('--example_count', '-ec', type=int, default=10,
        help='Number of samples per object')

    parser.add_argument('--synset_list', '-syn', type = str,
        help = "Path to synset list")

    parser.add_argument('--synset_id', '-sid', type = int,
        help = "Index of synset")

    args = parser.parse_args()

    print("################################")
    print("Saving network to: {}".format(args.out))
    print("##### Data #####")
    print("Using database: {}".format(args.database))
    print("Voxels : {0:d}x{0:d}x{0:d}".format(args.voxels))
    print("Image size : 1x{0:d}x{0:d}".format(args.image))
        
    print("##### Training #####")
    print("Batchsize : {}".format(args.batchsize))
    print("Epochs : {}".format(args.epoch))
    print("################################")
    sys.stdout.flush()

    # Set up a neural network to train

    arch = archs[args.arch]
    enc = arch.Encoder()
    dec = arch.Decoder()
    dis = arch.Discriminator()

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, enc)

        f_name = lambda m: ff.join(args.out, "{}_iter_{}.npz".format(m,args.iter))
        serializers.npz.load_npz(f_name("enc"), enc)
        serializers.npz.load_npz(f_name("dec"), dec)
        serializers.npz.load_npz(f_name("dis"), dis)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.000002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.000001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc, alpha = 2.0e-5)
    opt_dec = make_optimizer(dec, alpha = 2.0e-5)
    opt_dis = make_optimizer(dis, alpha = 1.0e-5)

    if args.synset_list is not None:
        assert(args.synset_id is not None)
        syn = (args.synset_list, args.synset_id)
    else:
        syn = None

    dataset = data.CrossModalDataset(
        args.database, "images", "voxels", in_size=args.example_count, rot=args.rot, 
        image_size = (args.image, args.image), synset = syn
    )    
        
    train_size = int(len(dataset)*args.ratio)
    print("DATASET SIZE: {}".format(len(dataset)))

    train_d, test_d = chainer.datasets.split_dataset_random(dataset, train_size)
    train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=args.loaderjob)
    test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize, n_processes=args.loaderjob)


    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec, 
            'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    # trainer.extend(extensions.snapshot(
    #     filename='snapshot_iter_{.updater.iteration}.npz'),
    #                trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
