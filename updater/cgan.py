#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable
from chainer import cuda
from chainer import function
from chainer.utils import type_check

import numpy as np

from updater import gan

class Updater(gan.Updater):


    def update_core(self):        
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')
        
        enc, dec, dis = self.enc, self.dec, self.dis
        xp = enc.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        assert(batch[0][1].shape[0] % 2 == 0)

        out_ch = int(batch[0][1].shape[0] / 2)

        # Changing to voxel space
        w_in = 256
        w_out = 64
        
        x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype(np.float32)
        t_out = xp.zeros((batchsize, out_ch, w_out, w_out, w_out)).astype(np.float32)
        d_in = xp.zeros((batchsize, out_ch, w_out, w_out, w_out)).astype(np.float32)

        for i in range(batchsize):
            x_in[i,:] = xp.asarray(batch[i][0])
            t_out[i,:] = xp.asarray(batch[i][1][0])
            d_in[i,:] = xp.asarray(batch[i][1][1])

        x_in = Variable(x_in)
        t_out = Variable(t_out)
        d_in = Variable(d_in)

        with chainer.using_config('train', True):

            # This will no longer work for pix-pix
            z = enc(x_in)
            x_out = dec(z)

            y_fake = dis(x_out, d_in)
            y_real = dis(t_out, d_in)

            update_dis, update_gen = self.check_dis(y_real, y_fake)

            if update_gen:
                enc_optimizer.update(self.loss_enc, enc, x_out, t_out, y_fake)
                for z_ in z:
                    z_.unchain_backward()
               
                dec_optimizer.update(self.loss_dec, dec, x_out, t_out, y_fake)
            else:
                print("Not updating gen")
            
            
            x_in.unchain_backward()
            x_out.unchain_backward()

            if update_dis :
                dis_optimizer.update(
                    self.loss_dis, dis, y_real, y_fake)
            else:
                print("Not updating disc")
