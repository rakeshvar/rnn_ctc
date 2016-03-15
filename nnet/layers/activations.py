import theano.tensor as tt


class Activation:
    """
    Defines a bunch of activations as callable classes.
    Useful for printing and specifying activations as strings.
    (via activation_by_name)
    """
    def __init__(self, fn, name):
        self.fn = fn
        self.name = name

    def __call__(self, *args):
        return self.fn(*args)

    def __str__(self):
        return self.name


activation_list = [
    tt.nnet.sigmoid,
    tt.nnet.softplus,
    tt.nnet.softmax,
    Activation(lambda x: x, 'linear'),
    Activation(lambda x: 1.7*tt.tanh(2 * x / 3), 'scaled_tanh'),
    Activation(lambda x: tt.maximum(0, x), 'relu'),
    Activation(lambda x: tt.tanh(x), 'tanh'),
] + [
    Activation(lambda x, i=i: tt.maximum(0, x) + tt.minimum(0, x) * i/100,
               'relu{:02d}'.format(i))
    for i in range(100)
]


def activation_by_name(name):
    """
    Get an activation function or callabe-class from its name.
    Activation Names
     sigmoid, softplus, softmax, linear, scaled_tanh, tanh,
     relu, relu00, relu01, ..., relu99
    :param string name:
    :return: callable activation
    """
    for act in activation_list:
        if name == str(act):
            return act
    else:
        raise NotImplementedError('Unknown Activation Specified: ' + name)