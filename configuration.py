import net
import updater
import chainer

archs = {   "cgan" : (net.cgan, updater.cgan, True),
            "gan" : (net.gan, updater.gan, False),
        }

optimizers = {  "adam" : chainer.optimizers.Adam,
                "sgd" : chainer.optimizers.SGD
             }