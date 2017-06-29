#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.enc, self.dec, self.dis = kwargs.pop('models')
        self.iteration = kwargs.pop('iteration')
        self.lams = kwargs.pop('lams')
        super(Updater, self).__init__(*args, **kwargs)


    def loss_gen(self, opt, x_out, t_out, y_out):
        xp = cuda.get_array_module(x_out)
        lam1, lam2 = self.lams
        norm = np.prod(y_out.data.shape)
        # print(x_out.debug_print())
        print("MEAN_Z > 0.5: {}".format(
            xp.sum(x_out.data > 0.5)/ np.prod(x_out.data.shape)))
        print("MAX_Z ALL: {}".format(x_out.data.max()))
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / norm
        loss = (loss_rec + loss_adv) / (lam1 + lam2)
        chainer.report({'loss': loss}, opt)
        return loss
        
        
    def loss_dis(self, dis, y_in, y_out, a=1, b=1):
        L1 = a*F.sum(F.softplus(-y_in)) / len(y_in) 
        L2 = b*F.sum(F.softplus(y_out)) /len(y_out)

        loss = (L1 + L2) / (a+b)
        chainer.report({'loss': loss}, dis)
        return loss

    def check_dis(self, y_in, y_out):

        real = np.mean(y_in.data) 
        fake = np.mean(y_out.data)

        print("ACC R {} | F {} ".format(real, fake))

        # c0 = real < abs(fake)
        c1 = fake > 0.0 or real < 0.0
        c2 = fake < -10.0 # and not c0
        
        print(c1,c2)

        update_dis = c1 and not c2
        update_gen =  c2

        # return update_dis, update_gen 
        return True, True

    def update_core(self):        
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')
        
        enc, dec, dis = self.enc, self.dec, self.dis
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]

        # Changing to voxel space
        w_in = 256
        w_out = 64
        
        x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
        t_out = xp.zeros((batchsize, out_ch, w_out, w_out, w_out)).astype("f")

        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1])
        x_in = Variable(x_in)
        t_out = Variable(t_out)


        with chainer.using_config('train', True):

            # This will no longer work for pix-pix
            z = enc(x_in)
            x_out = dec(z)

            y_fake = dis(x_out)
            y_real = dis(t_out)


            update_dis, update_gen = self.check_dis(y_real, y_fake)

            if update_gen:
                enc_optimizer.update(self.loss_gen, enc, x_out, t_out, y_fake)
                for z_ in z:
                    z_.unchain_backward()
               
                dec_optimizer.update(self.loss_gen, dec, x_out, t_out, y_fake)
            else:
                print("Not updating gen")
            
            
            x_in.unchain_backward()
            x_out.unchain_backward()

            if update_dis :
                dis_optimizer.update(
                    self.loss_dis, dis, y_real, y_fake)
            else:
                print("Not updating disc")
