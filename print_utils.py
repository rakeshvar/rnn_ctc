# -*- coding: utf-8 -*-


def slab_print(slab):
    """
    Prints a 'slab' of printed 'text' using ascii.
    :param slab: A matrix of floats from [0, 1]
    """
    for ir, r in enumerate(slab):
        print('{:2d}¦'.format(ir), end='')
        for val in r:
            if val   < .05:  print(' ', end=''),
            elif val < .35:  print('░', end=''),
            elif val < .65:  print('▒', end=''),
            elif val < .95:  print('▓', end=''),
            else:            print('█', end=''),
        print('¦')


def prediction_printer(n_classes):
    """
    Returns a function that can print a predicted output of the CTC RNN
    It removes the blank characters (need to be set to n_classes),
    It also removes duplicates
    :param n_classes: index of blank character
    :return: the printing function
    """

    def yprint(labels):
        for il, l in enumerate(labels):
            if (l != n_classes) and (il==0 or l != labels[il-1]):
                    print(l, end=' ')
        print()
    return yprint
