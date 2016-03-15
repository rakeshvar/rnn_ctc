# -*- coding: utf-8 -*-
import ast
import numpy as np
import scribe


def slab_print(slab, col_names=None):
    """
    Prints a 'slab' of printed 'text' using ascii.
    :param slab: A matrix of floats from [0, 1]
    """
    for ir, r in enumerate(slab):
        print('{:2d}¦'.format(ir), end='')
        for val in r:
            if   val < 0.0:  print('-', end='')
            elif val < .15:  print(' ', end=''),
            elif val < .35:  print('░', end=''),
            elif val < .65:  print('▒', end=''),
            elif val < .85:  print('▓', end=''),
            elif val <= 1.:  print('█', end=''),
            else:            print('+', end='')
        print('¦ {}'.format(col_names[ir] if col_names else ''))


class Printer():
    def __init__(self, chars):
        """
        Creates a function that can print a predicted output of the CTC RNN
        It removes the blank characters (need to be set to n_classes),
        It also removes duplicates
        :param list chars: list of characters
        """
        self.chars = chars + ['blank']
        self.n_classes = len(self.chars) - 1

    def yprint(self, labels):
        labels_out = []
        for il, l in enumerate(labels):
            if (l != self.n_classes) and (il == 0 or l != labels[il-1]):
                labels_out.append(l)
        labels_list = [self.chars[l] for l in labels_out]
        print(labels_out, ' '.join(labels_list))
        return labels_out, labels_list

    def ylen(self, labels):
        length = 0
        for il, l in enumerate(labels):
            if (l != self.n_classes) and (il == 0 or l != labels[il-1]):
                length += 1
        return length

    def show_all(self, shown_seq, shown_img,
                 softmax_firings=None,
                 *aux_imgs):
        """
        Utility function to show the input and output and debug
        :param shown_seq: Labelings of the input
        :param shown_img: Input Image
        :param softmax_firings: Seen Probabilities (Excitations of Softmax)
        :param aux_imgs: List of pairs of images and names
        :return:
        """
        print('Shown : ', end='')
        labels, labels_chars = self.yprint(shown_seq)

        if softmax_firings is not None:
            print('Seen  : ', end='')
            maxes = np.argmax(softmax_firings, 0)
            self.yprint(maxes)

        print('Image Shown:')
        slab_print(shown_img)

        if softmax_firings is not None:
            labels.append(self.n_classes)
            labels_chars.append('blank')
            print('SoftMax Firings:')
            slab_print(softmax_firings[labels], labels_chars)

        for aux_img, aux_name in aux_imgs:
            print(aux_name)
            slab_print(aux_img)


def insert_blanks(y, blank):
    # Insert blanks at alternate locations in the labelling (blank is blank)
    y1 = [blank]
    for char in y:
        y1 += [char, blank]
    return y1


def read_args(files, default='configs/default.ast'):
    with open(default, 'r') as dfp:
        args = ast.literal_eval(dfp.read())

    for config_file in files:
        with open(config_file, 'r') as cfp:
            override_args = ast.literal_eval(cfp.read())

        for key in args:
            if key in override_args:
                try:
                    args[key].update(override_args[key])
                except AttributeError:
                    args[key] = override_args[key]

    try:
        args['scribe_args']['alphabet'] = getattr(scribe, args['scribe_args']['alphabet'])
    except KeyError:
        pass

    return args