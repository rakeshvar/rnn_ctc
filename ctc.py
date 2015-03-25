import theano
import theano.tensor as tt
from activations import share, init_wts


class SoftmaxLayer():
    def __init__(self, inpt, in_sz, n_classes, ):
        b = share(init_wts(n_classes))
        w = share(init_wts(in_sz, n_classes))
        self.output = tt.nnet.softmax(tt.dot(inpt, w) + b)
        self.params = [w, b]


class CTCLayer():
    def __init__(self, inpt, labels, blank):
        """
        Recurrent Relation:
        A matrix that specifies allowed transistions in paths.
        At any time, one could
        0) Stay at the same label (diagonal is identity)
        1) Move to the next label (first upper diagonal is identity)
        2) Skip to the next to next label if
            a) next label is blank and
            b) the next to next label is different from the current
            (Second upper diagonal is product of conditons a & b)
        """
        labels2 = tt.concatenate((labels, [blank, blank]))
        sec_diag = tt.neq(labels2[:-2], labels2[2:]) * tt.eq(labels2[1:-1],
                                                             blank)
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
        labels_probab = tt.sum(probabilities[-1, -2:])
        self.cost = -tt.log(labels_probab)
        self.params = []
        self.debug = probabilities.T