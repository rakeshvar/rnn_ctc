import theano.tensor as tt

from .weights import init_wts, share


class LinearLayer():
    def __init__(self, inpt, in_sz, n_classes, tied=False):
        if tied:
            b = share(init_wts(n_classes-1))
            w = share(init_wts(in_sz, n_classes-1))
            w1 = tt.horizontal_stack(w, tt.zeros((in_sz, 1)))
            b1 = tt.concatenate((b, tt.zeros(1)))
            self.output = tt.dot(inpt, w1) + b1
        else:
            b = share(init_wts(n_classes))
            w = share(init_wts(in_sz, n_classes))
            self.output = tt.dot(inpt, w) + b
        self.params = [w, b]


class SoftmaxLayer(LinearLayer):
    def __init__(self, inpt, in_sz, n_classes, tied=False):
        super().__init__(inpt, in_sz, n_classes, tied)
        self.output = tt.nnet.softmax(self.output)


########################### Experimental Layers that do not work !


class TanhLayer(LinearLayer):
    def __init__(self, inpt, in_sz, n_classes, tied=False):
        super().__init__(inpt, in_sz, n_classes, tied)
        self.output = tt.tanh(self.output)


class HardMaxLayer(LinearLayer):
    def __init__(self, inpt, in_sz, n_classes, tied=False):
        super().__init__(inpt, in_sz, n_classes, tied)
        self.output -= tt.max(self.output, axis=0)


class MeanLayer(LinearLayer):
    def __init__(self, inpt, in_sz, n_classes, tied=False):
        super().__init__(inpt, in_sz, n_classes, tied)
        self.output -= tt.mean(self.output, axis=0)