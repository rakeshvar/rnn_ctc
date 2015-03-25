import numpy as np
import theano
import theano.tensor as tt
from activations import share, init_wts


class RecurrentLayer():
    def __init__(self, inpt, nin, nunits, conv_sz=1):
        # inpt is transposed a priori
        tablet_wd, _ = inpt.shape
        if conv_sz > 1:
            inpt_clipped = inpt[:conv_sz * (tablet_wd // conv_sz), :]
            inpt_conv = inpt_clipped.reshape(
                (tablet_wd // conv_sz, nin * conv_sz))
        else:
            inpt_conv = inpt

        wio = share(init_wts(nin * conv_sz, nunits))  # input to output
        woo = share(init_wts(nunits, nunits))  # output to output
        bo = share(init_wts(nunits))

        def step(in_t, out_tm1):
            return tt.tanh(tt.dot(out_tm1, woo) + tt.dot(in_t, wio) + bo)

        self.output, _ = theano.scan(
            step,
            sequences=[inpt_conv],
            outputs_info=[np.zeros(nunits)]
        )

        self.params = [wio, woo, bo]
        self.nout = nunits

    @classmethod
    def __str__(cls):
        return "RecurrentLayer"