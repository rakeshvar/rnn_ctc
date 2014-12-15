#!/usr/bin/env python3
'''
Recurrent Neural Net with Connectionist Temporal Classification implemented
in Theano.

'''
import theano
import theano.tensor as tt
import numpy as np


def init_wts(*argv):
    return 1 * (np.random.rand(*argv) - 0.5)


def share(array, dtype=theano.config.floatX, name=None):
    return theano.shared(value=np.asarray(array, dtype=dtype), name=name)


def build_recurrent_layer(inpt, wih, whh, bh, h0):
    def step(x_t, h_tm1):
        h_t = tt.tanh(tt.dot(h_tm1, whh) + tt.dot(x_t, wih) + bh)
        return h_t

    hidden, _ = theano.scan(
        step,
        sequences=[inpt],
        outputs_info=[h0]
    )
    return hidden


def build_model(input_sz, hidden_sz, output_sz):
    x_sym = tt.matrix('X')
    wih = share(init_wts(input_sz, hidden_sz))  # input to hidden
    whh = share(init_wts(hidden_sz, hidden_sz))  # hidden to hidden
    # whh is a matrix means all hidden units are fully inter-connected
    who = share(init_wts(hidden_sz, output_sz))  # hidden to output
    bh = share(init_wts(hidden_sz))
    h0 = share(init_wts(hidden_sz))
    bo = share(init_wts(output_sz))

    params = [wih, whh, who, bh, h0, bo]
    hidden = build_recurrent_layer(x_sym, wih, whh, bh, h0)

    predict = tt.nnet.softmax(tt.dot(hidden, who) + bo)
    return x_sym, predict, params


def recurrence_relation(size):
    eye2 = tt.eye(size + 2)
    return tt.eye(size) + eye2[2:, 1:-1] + eye2[2:, :-2] * (tt.arange(size) % 2)


def path_probabs(predict, y_sym):
    pred_y = predict[:, y_sym]
    rr = recurrence_relation(y_sym.shape[0])

    def step(p_curr, p_prev):
        return p_curr * tt.dot(p_prev, rr)

    probabilities, _ = theano.scan(
        step,
        sequences=[pred_y],
        outputs_info=[tt.eye(y_sym.shape[0])[0]]
    )
    return probabilities


def ctc_cost(predict, y_sym):
    forward_probabs = path_probabs(predict, y_sym)
    backward_probabs = path_probabs(predict[::-1], y_sym[::-1])[::-1, ::-1]
    probabs = forward_probabs * backward_probabs / predict[:, y_sym]
    total_probabs = tt.sum(probabs)
    return -tt.log(total_probabs)


class RnnCTC():
    def __init__(self, n_dims, n_hidden, n_classes):
        self.Y = tt.ivector('Y')
        self.X, self.predict, self.params = \
            build_model(n_dims, n_hidden, n_classes + 1)
        self.cost = ctc_cost(self.predict, self.Y)

        # Differentiable
        self.updates = []
        for p in self.params:
            grad = tt.grad(self.cost, wrt=p)
            self.updates.append((p, p - .001 * grad))

    def get_train_fn(self):
        return theano.function(
            inputs=[self.X, self.Y],
            outputs=[self.predict, self.cost],
            updates=self.updates, )

    def get_test_fn(self):
        return theano.function(
            inputs=[self.X],
            outputs=[self.predict], )


# ##########################################################################
def show_all(shown_seq, shown_img, seen_probabilities):
    maxes = np.argmax(seen_probabilities, 1)
    print('Shown : ', end='')
    pred_print(shown_seq)
    print('Seen  : ', end='')
    pred_print(maxes)
    print('Images (Shown & Seen) : ')
    slab_print(shown_img)
    slab_print(seen_probabilities.T)


if __name__ == "__main__":
    import pickle
    import sys
    from print_utils import slab_print, prediction_printer

    nHidden = 9

    if len(sys.argv) < 2:
        print('Usage\n{} <data_file.pkl> [nHidden={}]'
              ''.format(sys.argv[0], nHidden))
        sys.exit(1)

    with open(sys.argv[1], "rb") as pkl_file:
        data = pickle.load(pkl_file)

    try:
        nHidden = int(sys.argv[2])
    except IndexError:
        pass

    nClasses = data['nChars']
    nDims = len(data['x'][0])
    nSamples = len(data['x'])
    nTrainSamples = nSamples * .75
    nEpochs = 100

    ntwk = RnnCTC(nDims, nHidden, nClasses)
    train_fn = ntwk.get_train_fn()
    test_fn = ntwk.get_test_fn()
    pred_print = prediction_printer(nClasses)

    data_x, data_y = [], []
    for x, y in zip(data['x'], data['y']):
        # Need to make alternate characters blanks (index as nClasses)
        y1 = [nClasses]
        for char in y:
            y1 += [char, nClasses]
        data_y.append(np.asarray(y1, dtype=np.int32))
        data_x.append(np.asarray(x, dtype=theano.config.floatX))

    # Actual training
    for epoch in range(nEpochs):
        print('Epoch : ', epoch)
        for samp in range(nSamples):
            x = data_x[samp]
            y = data_y[samp]

            if samp < nTrainSamples:
                pred, cst = train_fn(x.T, y)
                if epoch % 10 == 0 and samp < 3:
                    print('## TRAIN cost: ', np.round(cst, 3))
                    show_all(y, x, pred)

            elif (epoch % 10 == 0 or epoch == nEpochs - 1) and \
                    samp - nTrainSamples < 3:
                # Print some test images
                pred = np.asarray(test_fn(x.T))[0]
                print('## TEST')
                show_all(y, x, pred)