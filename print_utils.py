# -*- coding: utf-8 -*-


def slab_print(slab):
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
        print('¦')


def prediction_printer(chars):
    """
    Returns a function that can print a predicted output of the CTC RNN
    It removes the blank characters (need to be set to n_classes),
    It also removes duplicates
    :param list chars: list of characters
    :return: the printing functions
    """
    n_classes = len(chars)

    def yprint(labels):
        labels_out = []
        for il, l in enumerate(labels):
            if (l != n_classes) and (il == 0 or l != labels[il-1]):
                labels_out.append(l)
        print(labels_out, " ".join(chars[l] for l in labels_out))

    def ylen(labels):
        length = 0
        for il, l in enumerate(labels):
            if (l != n_classes) and (il == 0 or l != labels[il-1]):
                length += 1
        return length

    return yprint, ylen
