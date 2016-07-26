import pickle
import sys
from time import clock

import numpy as np
import theano as th
import theano.tensor as tt

from configurations import configs
from neuralnet import NeuralNet
from print_utils import slab_print, prediction_printer

# th.config.optimizer = 'fast_compile'
# th.config.exception_verbosity='high'
th.config.profile = True


def show_all(shown_seq, shown_img,
             seen_probabilities=None,
             aux_img=None, aux_name=None):
    """
    Utility function to show the input and output and debug
    :param shown_seq: Labelings of the input
    :param shown_img: Input Image
    :param seen_probabilities: Seen Probabilities (Excitations of Softmax)
    :param aux_img: Other image/matrix for debugging
    :param aux_name: Name of aux
    :return:
    """
    print('Shown : ', end='')
    labels_print(shown_seq)

    if seen_probabilities is not None:
        print('Seen  : ', end='')
        maxes = np.argmax(seen_probabilities, 0)
        labels_print(maxes)

    print('Image Shown:')
    slab_print(shown_img)

    if seen_probabilities is not None:
        print('SoftMax Firings:')
        slab_print(seen_probabilities)

    if aux_img is not None:
        print(aux_name)
        slab_print(aux_img)

# ################################## Main Script ###########################
config_num = 0
log_space = True

if len(sys.argv) < 2:
    print('Usage'
          '\n\t{} <data_file.pkl> [configuration#={}] [use_log={}]'
          '\n\nConfigurations:'.format(sys.argv[0], config_num, log_space))
    for i, config in enumerate(configs):
        print("{:2d} {} {}".format(i, config[0].__name__, config[1]))
    print("Edit configurations.py to add new or change existing "
          "configurations.")
    sys.exit(1)

with open(sys.argv[1], "rb") as pkl_file:
    data = pickle.load(pkl_file)

if len(sys.argv) > 2:
    config_num = int(sys.argv[2])

if len(sys.argv) > 3:
    log_space = sys.argv[3][0] in "TtYy1"

################################
# Network Parameters

midlayer, midlayerargs = configs[config_num]
data_x, data_y, chars = data['x'], data['y'], data['chars']
del data

nClasses = len(chars)
nDims = len(data_x[0])
nSamples = len(data_x)
nTrainSamples = int(nSamples * .8)
nEpochs = 10
labels_print, labels_len = prediction_printer(chars)

print("\nConfig: {}"
      "\n   Midlayer: {} {}"
      "\nInput Dim: {}"
      "\nNum Classes: {}"
      "\nNum Samples: {}"
      "\nFloatX: {}"
      "\nUsing log space: {}"
      "\n".format(config_num, midlayer, midlayerargs, nDims, nClasses,
                  nSamples, th.config.floatX, log_space))

################################
print("Preparing the Data")
try:
    conv_sz = midlayerargs["conv_sz"]
except KeyError:
    conv_sz = 1

for i in range(nSamples):
    # Insert blanks at alternate locations in the labelling (blank is nClasses)
    y1 = [nClasses]
    for char in data_y[i]:
        y1 += [char, nClasses]

    data_y[i] = np.asarray(y1, dtype=np.int32)
    data_x[i] = data_x[i].astype(th.config.floatX)

    # Check to see if
    if labels_len(y1) > (1 + data_x[i].shape[1]) // conv_sz:
        bad_data = True
        show_all(y1, data_x[i], None, data_x[i][:, ::conv_sz], "Squissed")


def merge(d):
    indices = np.cumsum([0] + list((dd.shape[-1] for dd in d))).astype(
        'float32')
    merged = np.concatenate(d, axis=-1).astype('float32')
    print(merged.shape)
    return th.shared(merged, borrow=False), tt.cast(th.shared(indices), 'int32')

xs, x_indices = merge(data_x)
ys, y_indices = merge(data_y)
ys = tt.cast(ys, 'int32')
for s in xs, x_indices, ys, y_indices:
    print(s, type(s), )#s.get_value().shape)

################################
print("Building the Network")

ntwk = NeuralNet(nDims, nClasses, midlayer, midlayerargs,
                 xs, x_indices, ys, y_indices,
                 init_learn_rate=.1,
                 logspace=log_space)

print("Training the Network")
start = clock()
curr = clock()
for epoch in range(nEpochs):
    print('Epoch : {} Time: {:.2f}'.format(epoch, clock()-curr))
    curr = clock()
    ntwk.update_learning_rate(epoch)

    for samp in range(nSamples):
        x = data_x[samp]
        y = data_y[samp]
        # if not samp % 500:            print(samp)

        if samp < nTrainSamples:
            if log_space and len(y) < 2:
                continue

            if epoch % 10 == 0 and samp < 3:
                cost, pred, aux = ntwk.trainer_dbg(samp)
                print('\n## TRAIN cost: ', np.round(cost, 3))
                show_all(y, x, pred, aux > 1e-20, 'Forward probabilities:')
            else:
                cost = ntwk.trainer(samp)

            if np.isinf(cost) or np.isnan(cost):
                print('Exiting on account of Inf Cost...')
                sys.exit()

        elif (epoch % 10 == 0 and samp - nTrainSamples < 3) \
                or epoch == nEpochs - 1:
            # Print some test images
            pred, aux = ntwk.tester_dbg(samp)
            aux = (aux + 1) / 2.0
            print('\n## TEST')
            show_all(y, x, pred, aux, 'Hidden Layer:')

end = clock()
mins, secs = int(end-start)//60, (end-start)%60
print("Time taken for {} epochs is : {}m {:.2f}s".format(nEpochs, mins, secs))