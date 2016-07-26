import numpy as np
import theano as th
import theano.tensor as tt
from ctc import CTCLayer
from outlayers import SoftmaxLayer


def momentum_updates(params, cost, rate, momentum=.0):
    updates = []
    for param in params:
        accum_update = th.shared(param.get_value(borrow=True) * 0,
                                 broadcastable=param.broadcastable)
        curr_update = momentum * accum_update + (1 - momentum) * tt.grad(cost, param)
        updated_param = param - rate * accum_update
        updates.append((accum_update, curr_update))
        updates.append((param, updated_param))
    return updates

class NeuralNet():
    def __init__(self, n_dims, n_classes,
                 midlayer, midlayer_args,
                 x, x_indices,
                 y, y_indices,
                 init_learn_rate,
                 logspace):
        image = tt.matrix('image')
        labels = tt.ivector('labels')

        layer1 = midlayer(image.T, n_dims, **midlayer_args)
        layer2 = SoftmaxLayer(layer1.output, layer1.nout, n_classes + 1)
        layer3 = CTCLayer(layer2.output, labels, n_classes, logspace)

        self.init_learn_rate = init_learn_rate
        self.learning_rate = th.shared(np.cast[th.config.floatX](0.0))
        params = (p for lyr in (layer1, layer2, layer3) for p in lyr.params)
        updates = momentum_updates(params, layer3.cost, self.learning_rate)
        self.update_learning_rate(0)

        indx = tt.lscalar('index')
        givens_tr = {
            image: x[:,x_indices[indx]:x_indices[indx+1]],
            labels:  y[y_indices[indx]:y_indices[indx+1]]}

        self.trainer = th.function([indx], layer3.cost,
            givens=givens_tr, updates=updates)

        self.trainer_dbg = th.function([indx],
            [layer3.cost, layer2.output.T, layer3.debug],
            givens=givens_tr, updates=updates)

        givens_te={image:x[:,x_indices[indx]:x_indices[indx+1]]}

        # self.tester = th.function([indx], layer2.output.T, givens=givens_te)

        self.tester_dbg = th.function([indx],
            [layer2.output.T, layer1.output.T],
            givens=givens_te)

    def update_learning_rate(self, epoch):
        self.learning_rate.set_value(self.init_learn_rate/(1 + epoch))