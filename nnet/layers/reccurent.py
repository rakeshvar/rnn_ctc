import theano
import theano.tensor as tt

from .weights import init_wts, share


class RecurrentLayer():
    def __init__(self, inpt, nin, nunits, conv_sz=1,
                 learn_init_state=True):
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
        h0 = share(init_wts(nunits))

        def step(in_t, out_tm1):
            return tt.tanh(tt.dot(out_tm1, woo) + tt.dot(in_t, wio) + bo)

        self.output, _ = theano.scan(
            step,
            sequences=[inpt_conv],
            outputs_info=[h0]
        )

        self.params = [wio, woo, bo]
        if learn_init_state:
            self.params += [h0]
        self.nout = nunits


class BiRecurrentLayer():
    def __init__(self, inpt, nin, nunits, conv_sz=1,
                 learn_init_state=True):
        fwd = RecurrentLayer(inpt, nin, nunits, conv_sz, learn_init_state)
        bwd = RecurrentLayer(inpt[::-1], nin, nunits, conv_sz, learn_init_state)

        self.params = fwd.params + bwd.params
        self.nout = fwd.nout + bwd.nout
        self.output = tt.concatenate([fwd.output,
                                      bwd.output[::-1]],
                                     axis=1)