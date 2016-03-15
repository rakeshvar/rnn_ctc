import pickle
import sys
import numpy as np
import theano as th
from nnet.neuralnet import NeuralNet
import utils

# th.config.optimizer = 'fast_compile'
# th.config.exception_verbosity='high'

################################### Main Script ###########################
print('Loading the dataset.')
with open(sys.argv[1], 'rb') as pkl_file:
    data = pickle.load(pkl_file)

args = utils.read_args(sys.argv[2:])
num_epochs, train_on_fraction = args['num_epochs'], args['train_on_fraction']
scribe_args, nnet_args,  = args['scribe_args'], args['nnet_args'],

chars = data['chars']
num_classes = len(chars)
img_ht = len(data['x'][0])
num_samples = len(data['x'])
nTrainSamples = int(num_samples * train_on_fraction)
printer = utils.Printer(chars)

print('\nInput Dim: {}'
      '\nNum Classes: {}'
      '\nNum Samples: {}'
      '\nNum Epochs: {}'
      '\nFloatX: {}'
      '\n'.format(img_ht, num_classes, num_samples, num_epochs, th.config.floatX))

################################
print('Preparing the Data')
try:
    conv_sz = nnet_args['midlayerargs']['conv_sz']
except KeyError:
    conv_sz = 1

data_x, data_y = [], []
bad_data = False

for x, y in zip(data['x'], data['y']):
    # Insert blanks at alternate locations in the labelling (blank is num_classes)
    y1 = utils.insert_blanks(y, num_classes)
    data_y.append(np.asarray(y1, dtype=np.int32))
    data_x.append(np.asarray(x, dtype=th.config.floatX))

    if printer.ylen(y1) > (1 + len(x[0])) // conv_sz:
        bad_data = True
        printer.show_all(y1, x, None, (x[:, ::conv_sz], 'Squissed'))


################################
print('Building the Network')
ntwk = NeuralNet(img_ht, num_classes, **nnet_args)
print(ntwk)

print('Training the Network')
for epoch in range(num_epochs):
    print('Epoch : ', epoch)
    for samp in range(num_samples):
        x = data_x[samp]
        y = data_y[samp]
        # if not samp % 500:            print(samp)

        if samp < nTrainSamples:
            if len(y) < 2:
                continue

            cst, pred, aux = ntwk.trainer(x, y)

            if (epoch % 10 == 0 and samp < 3) or np.isinf(cst):
                print('\n## TRAIN cost: ', np.round(cst, 3))
                printer.show_all(y, x, pred, (aux > 1e-20, 'Forward probabilities:'))

            if np.isinf(cst):
                print('Exiting on account of Inf Cost...')
                sys.exit()

        elif (epoch % 10 == 0 and samp - nTrainSamples < 3) \
                or epoch == num_epochs - 1:
            # Print some test images
            pred, aux = ntwk.tester(x)
            aux = (aux + 1) / 2.0
            print('\n## TEST')
            printer.show_all(y, x, pred, (aux, 'Hidden Layer:'))

print(ntwk)