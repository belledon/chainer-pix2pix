#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

# U-net https://arxiv.org/pdf/1611.07004v1.pdf

# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)

        elif sample=="dis":
            layers['c'] = L.ConvolutionND(3, ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.DeconvolutionND(3, ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)
        
    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h
    
class Encoder(chainer.Chain):
    def __init__(self, in_ch=1):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(256, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(256, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(256, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=True)
        layers['c7'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=True)
        layers['c8'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=True)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))

        # print(x.debug_print())
        # print("\n")
        # print("At layer {}".format(0))
        # print(h.debug_print())
        for i in range(1,9):

            
            # print("At layer {}".format(i))
            h = self['c{0:d}'.format(i)](h)
            # print(h.debug_print())
        return F.leaky_relu(h)

class Decoder(chainer.Chain):
    def __init__(self, out_ch=1):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c1'] = CBR(512, 256, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c2'] = CBR(256, 256, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c3'] = CBR(256, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(256, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(128, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = L.DeconvolutionND(3, 128, 64, 3, 1, 1, initialW=w)
        layers['c7'] = L.DeconvolutionND(3, 64, out_ch, 3, 1, 1, initialW=w)
        super(Decoder, self).__init__(**layers)

    def __call__(self, x):

        x = F.expand_dims(x, axis=-1)

        # print(x.debug_print())
        # print("\n")
        # print("At layer {}".format(0))
        h = self.c0(x)
        # print(h.debug_print())
        for i in range(1,8):
            
            h = self['c{0:d}'.format(i)](h)
            if i == 6:
                h = F.relu(h)
            # print("At layer {}".format(i))
        
        # print(h.debug_print())
            
        
        return F.sigmoid(h)



class Discriminator(chainer.Chain):
    def __init__(self, in_ch=1, out_ch=1):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_i'] = L.ConvolutionND(3, in_ch, 64, 4, 2, 1, initialW=w)
        layers['c0_c'] = L.ConvolutionND(3, in_ch, 64, 4, 2, 1, initialW=w)
        layers['c1'] = CBR(128, 128, bn=True, sample='dis', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='dis', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='dis', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='dis', activation=F.leaky_relu, dropout=False)
        layers['c5'] = L.ConvolutionND(3, 512, 1, 4, 2, 1, initialW=w)
        super(Discriminator, self).__init__(**layers)

    def __call__(self, x, c):

        h_i = F.leaky_relu(self.c0_i(x))
        h_c = F.leaky_relu(self.c0_c(c))
        h = F.concat([h_i, h_c])
        
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = self.c5(h)
        
        return h

