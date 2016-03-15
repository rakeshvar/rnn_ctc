import theano.tensor as tt
import theano as th
from theano.tensor.nnet import sigmoid
import numpy as np

from .activations import activation_by_name
from .weights import stacked_ortho_wts, share


class LSTM():
    """
    Long Short Term Memory Layer.
    Does not implement incell connections from cell value to the gates.
    Reference: Supervised Sequence Learning with RNNs by Alex Graves
                Chapter 4, Fig 4.2
    """
    def __init__(self, inpt,
                 nin, nunits,
                 forget=False,
                 actvn_pre='tanh',
                 actvn_post='linear',
                 learn_init_states=True):
        """
        Init
        :param inpt: Lower layer's excitation.
        :param nin: Dimension of lower layer.
        :param nunits: Number of units.
        :param forget: Want a seperate forget gate (or use 1-input)?
        :param actvn_pre: Activation applied to new candidate for cell value.
        :param actvn_post: Activation applied to cell value before output.
        :param learn_init_states: Should the intial states be learnt?
        :return: Output
        """
        # TODO: Incell connections

        num_activations = 3 + forget
        w = stacked_ortho_wts(nin, nunits, num_activations)
        u = stacked_ortho_wts(nunits, nunits, num_activations)
        b = share(np.zeros(num_activations * nunits))
        out0 = share(np.zeros(nunits))
        cell0 = share(np.zeros(nunits))

        actvn_pre = activation_by_name(actvn_pre)
        actvn_post = activation_by_name(actvn_post)

        def step(in_t, out_tm1, cell_tm1):
            """
            Scan function.
            :param in_t: Current input from bottom layer
            :param out_tm1: Prev output of LSTM layer
            :param cell_tm1: Prev cell value
            :return: Current output and cell value
            """
            tmp = tt.dot(out_tm1, u) + in_t

            inn_gate = sigmoid(tmp[:nunits])
            out_gate = sigmoid(tmp[nunits:2 * nunits])
            fgt_gate = sigmoid(
                tmp[2 * nunits:3 * nunits]) if forget else 1 - inn_gate

            cell_val = actvn_pre(tmp[-nunits:])
            cell_val = fgt_gate * cell_tm1 + inn_gate * cell_val
            out = out_gate * actvn_post(cell_val)

            return out, cell_val

        inpt = tt.dot(inpt, w) + b
        # seqlen x nin * nin x 3*nout + 3 * nout  = seqlen x 3*nout

        rval, updates = th.scan(step,
                                sequences=[inpt],
                                outputs_info=[out0, cell0], )

        self.output = rval[0]
        self.params = [w, u, b]
        if learn_init_states:
            self.params += [out0, cell0]
        self.nout = nunits


class BDLSTM():
    """
    Bidirectional Long Short Term Memory Layer.
    """
    def __init__(self, inpt,
                 nin, nunits,
                 forget=False,
                 actvn_pre='tanh',
                 actvn_post='linear',
                 learn_init_states=True):
        """
        See `LSTM`.
        """
        fwd = LSTM(inpt, nin, nunits, forget, actvn_pre, actvn_post,
                   learn_init_states)
        bwd = LSTM(inpt[::-1], nin, nunits, forget, actvn_pre, actvn_post,
                   learn_init_states)

        self.params = fwd.params + bwd.params
        self.nout = fwd.nout + bwd.nout
        self.output = tt.concatenate([fwd.output,
                                      bwd.output[::-1]],
                                     axis=1)