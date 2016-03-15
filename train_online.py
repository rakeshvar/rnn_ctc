import sys
import numpy as np
import theano as th
import utils
import nnet.neuralnet as nn
from scribe.scribe import Scribe

################################ Initialize
args = utils.read_args(sys.argv[1:])
num_samples, num_epochs = args['num_samples'], args['num_epochs']
scribe_args, nnet_args = args['scribe_args'], args['nnet_args']

print('\nArguments:'
      '\nFloatX         : {}'
      '\nNum Epochs     : {}'
      '\nNum Samples    : {}'
      '\n'.format(th.config.floatX, num_epochs, num_samples))

scriber = Scribe(**scribe_args)
printer = utils.Printer(scriber.alphabet.chars)
print(scriber)

print('Building the Network')
ntwk = nn.NeuralNet(scriber.nDims, scriber.nClasses, **nnet_args)
print(ntwk)

################################
print('Training the Network')

for epoch in range(num_epochs):
    ntwk.update_learning_rate(epoch)

    for samp in range(num_samples):
        x, y1 = scriber.get_sample()
        if len(y1) < 2:
            continue
        y = utils.insert_blanks(y1, scriber.nClasses)
        cst, pred, forward_probs = ntwk.trainer(x, y)

        if np.isinf(cst):
            printer.show_all(y, x, pred, (forward_probs > 1e-20, 'Forward probabilities:'))
            print('Exiting on account of Inf Cost...')
            break

        if samp == 0:
            pred, hidden = ntwk.tester(x)

            print('Epoch:{:6d} Cost:{:.3f}'.format(epoch, float(cst)))
            printer.show_all(y, x, pred,
                             (forward_probs > 1e-20, 'Forward probabilities:'),
                             ((hidden + 1)/2, 'Hidden Layer:'))