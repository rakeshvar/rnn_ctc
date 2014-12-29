#!/usr/bin/env python3
'''
Recurrent Neural Net with Connectionist Temporal Classification implemented
in Theano.

'''
import theano
import theano.tensor as tt
import numpy as np
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity='high'

def init_wts(*argv):
    return 1 * (np.random.rand(*argv) - 0.5)

def share(array, dtype=theano.config.floatX, name=None):
    return theano.shared(value=np.asarray(array, dtype=dtype), name=name)

class RecurrentLayer():
    def __init__(self, inpt, in_sz, out_sz, conv_sz=1):
        tablet_wd, _ = inpt.shape         # Remember X is transposed before being input
        if conv_sz > 1:
            inpt_clipped = inpt[:conv_sz*(tablet_wd//conv_sz), :]
            inpt_conv = inpt_clipped.reshape((tablet_wd//conv_sz, in_sz*conv_sz))
        else:
            inpt_conv = inpt

        wio = share(init_wts(in_sz*conv_sz, out_sz))  # input to output
        woo = share(init_wts(out_sz, out_sz))      # output to output
        bo  = share(init_wts(out_sz))

        def step(in_t, out_tm1):
            return tt.tanh(tt.dot(out_tm1, woo) + tt.dot(in_t, wio) + bo)

        self.output, _ = theano.scan(
            step,
            sequences=[inpt_conv],
            outputs_info=[np.zeros(out_sz)]
        )
        
        self.params = [wio, woo, bo]


class SoftmaxLayer():
    def __init__(self, inpt, in_sz, n_classes,):
        b = share(init_wts(n_classes))
        w = share(init_wts(in_sz, n_classes))
        self.output = tt.nnet.softmax(tt.dot(inpt, w) + b)
        self.params = [w, b]


class CTCLayer():
    def __init__(self, inpt, labels, blank):
        '''
        Recurrent Relation:
        A matrix that specifies allowed transistions in paths.
        At any time, one could 
        0) Stay at the same label (diagonal is identity)
        1) Move to the next label (first upper diagonal is identity)
        2) Skip to the next to next label if 
            a) next label is blank and 
            b) the next to next label is different from the current
            (Second upper diagonal is product of conditons a & b)
        '''
        labels2 = tt.concatenate((labels, [blank, blank]))
        sec_diag = tt.neq(labels2[:-2], labels2[2:]) * tt.eq(labels2[1:-1], blank)
        n_labels = labels.shape[0]

        recurrence_relation = \
               tt.eye(n_labels) + \
               tt.eye(n_labels, k=1) + \
               tt.eye(n_labels, k=2) * sec_diag.dimshuffle((0, 'x'))

        '''
        Forward path probabilities
        '''
        pred_y = inpt[:, labels]

        probabilities, _ = theano.scan(
            lambda curr, prev: curr * tt.dot(prev, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[tt.eye(n_labels)[0]]
        )

        # Final Costs
        labels_probab = tt.sum(probabilities[-1,-2:])
        self.cost = -tt.log(labels_probab)
        self.params = []
        self.debug = probabilities.T

class RnnCTC():
    def __init__(self, n_dims, n_hidden, n_classes, conv_sz):
        image = tt.matrix('image')
        labels = tt.ivector('labels')
        
        layer1 = RecurrentLayer(image.T, n_dims, n_hidden, conv_sz)
        layer2 = SoftmaxLayer(layer1.output, n_hidden, n_classes+1)
        layer3 = CTCLayer(layer2.output, labels, n_classes)


        updates = []
        for lyr in (layer1, layer2, layer3):
            for p in lyr.params:
                grad = tt.grad(layer3.cost, p)
                updates.append((p, p - .001 * grad))

        self.trainer = theano.function(
            inputs=[image, labels],
            outputs=[layer3.cost, layer2.output.T, layer3.debug],
            updates=updates, )

        self.tester = theano.function(
            inputs=[image],
            outputs=[layer2.output.T, layer1.output.T], )


# ##########################################################################
def show_all(shown_seq, shown_img, seen_probabilities, aux, aux_name):
    maxes = np.argmax(seen_probabilities, 0)
    print('Shown : ', end='')
    pred_print(shown_seq)
    print('Seen  : ', end='')
    pred_print(maxes)
    print('Images (Shown & Seen) : ')
    slab_print(shown_img)
    slab_print(seen_probabilities)
    print(aux_name)
    slab_print(aux)


if __name__ == "__main__":
    import pickle
    import sys
    from print_utils import slab_print, prediction_printer

    conv_sz = 3
    nHidden = 9

    if len(sys.argv) < 2:
        print('Usage\n{} <data_file.pkl> [conv_sz={}] [nHidden={}]'
              ''.format(sys.argv[0], conv_sz, nHidden))
        sys.exit(1)

    with open(sys.argv[1], "rb") as pkl_file:
        data = pickle.load(pkl_file)

    if len(sys.argv) > 2:
        conv_sz = int(sys.argv[2])

    if len(sys.argv) > 3:
        nHidden = int(sys.argv[3])

    nClasses = data['nChars']
    nDims = len(data['x'][0])
    nSamples = len(data['x'])
    nTrainSamples = nSamples * .75
    nEpochs = 100

    pred_print = prediction_printer(nClasses)

    data_x, data_y = [], []
    bad_data = False
    for x, y in zip(data['x'], data['y']):
        # Need to make alternate characters blanks (index as nClasses)
        y1 = [nClasses]
        for char in y:
            y1 += [char, nClasses]
        data_y.append(np.asarray(y1, dtype=np.int32))
        data_x.append(np.asarray(x, dtype=theano.config.floatX))
        if len(y1) > (1+len(x[0]))//conv_sz:
            bad_data = True
            show_all(y1, x, x[:2,::conv_sz].T)
    if bad_data: 
        print('BAD DATA')
        #sys.exit()

    ntwk = RnnCTC(nDims, nHidden, nClasses, conv_sz)
    # Actual training
    for epoch in range(nEpochs):
        print('Epoch : ', epoch)
        for samp in range(nSamples):
            x = data_x[samp]
            y = data_y[samp]

            if samp < nTrainSamples:
                cst, pred, aux = ntwk.trainer(x, y)
                if (epoch % 10 == 0 and samp < 3) or np.isinf(cst):
                    print('## TRAIN cost: ', np.round(cst, 3))
                    show_all(y, x, pred, aux>0, 'Forward probabilities:')
                if np.isinf(cst):
                    print('Exiting on account of Inf Cost on the following data...')
                    sys.exit()

            elif (epoch % 10 == 0 and samp - nTrainSamples < 3) \
                    or epoch == nEpochs - 1:
                # Print some test images
                pred, aux = ntwk.tester(x)
                aux = (aux + 1)/2.0
                print('## TEST')
                show_all(y, x, pred, aux, 'Convolution:')

