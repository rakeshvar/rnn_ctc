import theano as th
import theano.tensor as tt
from ctc import CTCLayer
from outlayers import SoftmaxLayer


class NeuralNet():
    def __init__(self, n_dims, n_classes,
                 midlayer, midlayer_args,
                 logspace=True):
        image = tt.matrix('image')
        labels = tt.ivector('labels')

        layer1 = midlayer(image.T, n_dims, **midlayer_args)
        layer2 = SoftmaxLayer(layer1.output, layer1.nout, n_classes + 1)
        layer3 = CTCLayer(layer2.output, labels, n_classes, logspace)

        updates = []
        for lyr in (layer1, layer2, layer3):
            for p in lyr.params:
                grad = tt.grad(layer3.cost, p)
                updates.append((p, p - .001 * grad))

        self.trainer = th.function(
            inputs=[image, labels],
            outputs=[layer3.cost, layer2.output.T, layer3.debug],
            updates=updates, )

        self.tester = th.function(
            inputs=[image],
            outputs=[layer2.output.T, layer1.output.T], )