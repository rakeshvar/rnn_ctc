import theano
import theano.tensor as tt

####################### Log Space Helpers ################################

eps, epsinv = 1e-20, 1e20

def safe_log(x):
    return tt.log(tt.maximum(x, eps).astype(theano.config.floatX))

def safe_exp(x):
    return tt.exp(tt.minimum(x, epsinv).astype(theano.config.floatX))

def logadd_simple(x, y):
    return x + safe_log(1 + safe_exp(y - x))

def logadd_advanced(x, y):
    maxx = tt.maximum(x, y)
    minn = tt.minimum(x, y)
    return maxx + tt.log(1 + tt.exp(minn - maxx))

def logadd(x, y, *zs, add=logadd_simple):
    sum = add(x, y)
    for z in zs:
        sum = add(sum, z)
    return sum

def logmul(x, y):
    return x + y


####################### Two Kinds of CTC Layers ################################
"""
Recurrent Relation:
    Specifies allowed transistions in paths.

    Implemented as
        Matrix in PlainCTC
        Masks in LogCTC

    At any time, one could feed in from the
        0) same label
            - diagonal is identity (Plain)
        1) prev label (unless ofcourse you are the first)
            - first upper diagonal is identity (Plain)
            - prevmask is [0, 1, 1, ..., 1] (Log)
        2) prev to prev label if
            a) next label is blank and
            b) the next to next label is different from the current
            - second_diag/prevprev_mask is product of conditions a & b
"""


class CTCLayer():
    def __init__(self, inpt, labels, blank, log_space):
        """
        :param inpt: Output of Soft-max layer
        :param labels: desired/correct labels
        :param blank: index of blank
        :param log_space: If calcualtions should be done in log space
        :return: CTCLayer object
        """
        self.inpt = inpt
        self.labels = labels
        self.blank = blank
        self.n = self.labels.shape[0]
        if log_space:
            self.log_ctc()
        else:
            self.plain_ctc()
        self.params = []

    def plain_ctc(self, ):
        labels2 = tt.concatenate((self.labels, [self.blank, self.blank]))
        sec_diag = tt.neq(labels2[:-2], labels2[2:]) * \
                   tt.eq(labels2[1:-1], self.blank)

        recurrence_relation = \
            tt.eye(self.n) + \
            tt.eye(self.n, k=1) + \
            tt.eye(self.n, k=2) * sec_diag.dimshuffle((0, 'x'))

        pred_y = self.inpt[:, self.labels]

        probabilities, _ = theano.scan(
            lambda curr, accum: curr * tt.dot(accum, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[tt.eye(self.n)[0]]
        )

        # TODO: -2 only if blank at end
        labels_probab = tt.sum(probabilities[-1, -2:])
        self.cost = -tt.log(labels_probab)
        self.debug = probabilities.T

    def log_ctc(self, ):
        _1000 = tt.eye(self.n)[0]
        prev_mask = 1 - _1000
        prevprev_mask = tt.neq(self.labels[:-2], self.labels[2:]) * \
                        tt.eq(self.labels[1:-1], self.blank)
        prevprev_mask = tt.concatenate(([0, 0], prevprev_mask))
        prev_mask = safe_log(prev_mask)
        prevprev_mask = safe_log(prevprev_mask)
        prev = tt.arange(-1, self.n-1)
        prevprev = tt.arange(-2, self.n-2)
        log_pred_y = tt.log(self.inpt[:, self.labels])

        def step(curr, accum):
            return logmul(curr,
                          logadd(accum,
                                 logmul(prev_mask, accum[prev]),
                                 logmul(prevprev_mask, accum[prevprev])))

        log_probs, _ = theano.scan(
            step,
            sequences=[log_pred_y],
            outputs_info=[safe_log(_1000)]
        )

        # TODO: Add -2 if n > 1 and blank at end
        log_labels_probab = log_probs[-1, -1]
        self.cost = -log_labels_probab
        self.debug = tt.exp(log_probs.T)