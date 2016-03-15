import numpy as np
import theano as th
import theano.tensor as tt

from .ctc import CTCLayer
from . import layers
from . import updates

class NeuralNet():
    def __init__(self, n_dims, n_classes,
                 mid_layer, mid_layer_args,
                 use_log_space=True,
                 optimizer='sgd', optimizer_args=None,
                 learning_rate_args=None,
                 ):
        image = tt.matrix('image')
        labels = tt.ivector('labels')

        # Layers
        mid_layer_type = getattr(layers, mid_layer)
        layer1 = mid_layer_type(image.T, n_dims, **mid_layer_args)
        layer2 = layers.SoftmaxLayer(layer1.output, layer1.nout, n_classes + 1)
        layer3 = CTCLayer(layer2.output, labels, n_classes, use_log_space)

        # Optimization
        params = [p for lyr in (layer1, layer2, layer3) for p in lyr.params]
        optimizer_type = getattr(updates, optimizer)
        learning_rate = th.shared(np.asarray(0, dtype=th.config.floatX))
        updated = optimizer_type(layer3.cost, params, learning_rate, **optimizer_args)

        self.trainer = th.function(
            inputs=[image, labels],
            outputs=[layer3.cost, layer2.output.T, layer3.debug],
            updates=updated, )

        self.tester = th.function(
            inputs=[image],
            outputs=[layer2.output.T, layer1.output.T], )

        self.learning_rate = learning_rate
        self.learning_rate_args = learning_rate_args
        self.update_learning_rate(0)

        self.repr = ('Neural Network:'
                 '\n Input Dimensions:{}'
                 '\n Number of output classes:{}'
                 '\n Use log space:{}'
                 '\n Middle layer type:{} ({})'
                 '\n Optimizer:{} ({})'
                 '\n Learning Rate:{}'
                 ''.format(n_dims, n_classes, use_log_space, mid_layer, mid_layer_args,
                           optimizer, optimizer_args, learning_rate_args))

    def update_learning_rate(self, epoch):
        a = self.learning_rate_args
        annealing = a['anneal']
        init_rate = a['initial_rate']
        epochs_to_half = a['epochs_to_half']

        if annealing == 'constant':
            new_value = init_rate
        elif annealing == 'inverse':
            new_value = init_rate / (1 + epoch / epochs_to_half)
        elif annealing == 'inverse_sqrt':
            new_value = init_rate / np.sqrt(1 + epoch / epochs_to_half)
        else:
            raise ValueError('Unknown Annealing', annealing)

        self.learning_rate.set_value(new_value)

    def __repr__(self):
        return self.repr