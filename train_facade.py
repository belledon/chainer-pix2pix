#!/usr/bin/env python


from __future__ import print_function
import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers


import fileFuncs as ff 

import data
import net
import updater

archs = {   "voxel-voxel" : (net.voxeltovoxel, updater.cgan, True),
            "pix-voxel" : (net.pixtovoxel, updater.gan, False),
        }

optimizers = {  "adam" : chainer.optimizers.Adam,
                "sgd" : chainer.optimizers.SGD
             }

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')


    parser.add_argument("database", type = str, 
        help = "Path for database")

    parser.add_argument("arch", type = str, choices = archs.keys(),
        help = "Network Architecture")

    parser.add_argument("--rot", "-r", type = int, default = 0,
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

    parser.add_argument("--resume", "-re", type = int, default = 0,
        help = "If > 0, load given iteration from ouput directory")

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

    parser.add_argument("--lams", '-l', type = int, default = 100,
        help = "Weights to apply to generator loss. Default (100, 1)")

    parser.add_argument("-opt", "--optimizer", type = str, choices = optimizers.keys(),
        default = "sgd", help = "Optimizer to train network. Default sgd")

    parser.add_argument("--gen_learn", "-gl", type = float, default = 2.0E-4,
        help = "Generator learning rate")

    parser.add_argument("--dis_learn", "-dl", type = float, default = 2.0E-4,
        help = "Discriminator learning rate")

    args = parser.parse_args()

    print("################################")
    print("Saving network to: {}".format(args.out))
    print("##### Data #####")
    print("Architecture : {}".format(args.arch))
    print("Using database: {}".format(args.database))
    print("Voxels : {0:d}x{0:d}x{0:d}".format(args.voxels))
    print("Image size : 1x{0:d}x{0:d}".format(args.image))
    print("Rotating: {} [0: No, 1: Yes]".format(args.rot))
    print("##### Training #####")
    print("OPTIMIZER : {}".format(args.optimizer))
    print("LAM1 : {}".format(args.lams))
    print("LAM2 : 1")
    print("GEN_LR : {}".format(args.gen_learn))
    print("DIS_LR: {}".format(args.dis_learn))
    print("Batchsize : {}".format(args.batchsize))
    print("Epochs : {}".format(args.epoch))
    print("################################")
    sys.stdout.flush()

    # Set up a neural network to train

    arch, arch_updater, cgan = archs[args.arch]
    enc = arch.Encoder()
    dec = arch.Decoder()
    dis = arch.Discriminator()


    if args.resume > 0:
        # Resume from a snapshot
        print("Resuming training from {} at iteration {}".format(args.out, args.resume))
        f_name = lambda m: ff.join(args.out, "{}_iter_{}.npz".format(m,args.resume))
        serializers.npz.load_npz(f_name("enc"), enc)
        serializers.npz.load_npz(f_name("dec"), dec)
        serializers.npz.load_npz(f_name("dis"), dis)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()



    # Setup an optimizer
    optimizer = optimizers[args.optimizer]
    
    def make_optimizer(model, lr, optimizer=optimizer):
        optimizer = optimizer(lr)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.000001), 'hook_dec')
        return optimizer

    opt_enc = make_optimizer(enc, args.gen_learn)
    opt_dis = make_optimizer(dis, args.dis_learn)
    opt_dec = make_optimizer(dec, args.gen_learn)


    if args.synset_list is not None:
        assert(args.synset_id is not None)
        syn = (args.synset_list, args.synset_id)
    else:
        syn = None

    dataset = data.CrossModalDataset(
        args.database, "images", "voxels", in_size=args.example_count, rot=args.rot, 
        image_size = (args.image, args.image), synset = syn, cgan=cgan
    )    
        
    train_size = int(len(dataset)*args.ratio)
    print("DATASET SIZE: {}".format(len(dataset)))

    train_d, test_d = chainer.datasets.split_dataset_random(dataset, train_size)
    train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=args.loaderjob)
    test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize, n_processes=args.loaderjob)


    # Set up a trainer
    up = arch_updater.FacadeUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec, 
            'dis': opt_dis},
        device=args.gpu,
        iteration = args.resume,
        lams = (args.lams, 1)
        )

    trainer = training.Trainer(up, (args.epoch, 'epoch'), out=args.out)

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
