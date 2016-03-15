#!/usr/bin/env python3
'''
This file is used to 'scribe' a random piece of 'text' on to a 'slab'.
  'text' - A sequence based on an alphabet [0, 1, 2 ...n_chars)
  'slab' - An numpy matrix
            Has as many rows as the size of the alphabet i.e. n_chars

  A character 'i' in the text is of length i+2 by default and is written in
  the i-th line! i.e. 2 is written in the 2nd line with a length of 4 'pixels'.

  Characters are read from left to right.

  Example:
  The text [2, 1, 3, 0] is written as
   0¦       ██ ¦
   1¦   ███    ¦
   2¦ ████     ¦
   3¦     █████¦
'''
import numpy as np


class RowScribe():
    def __init__(self, n_chars, avg_seq_len, buffer_len=3, char_lens=None):
        self.nChars = n_chars
        self.nDims = n_chars
        self.len = avg_seq_len
        self.buffer = buffer_len
        if char_lens is None:
            self.char_lens = np.arange(n_chars) + 2
        else:
            self.char_lens = None

    def get_data(self, complexx, vary):
        if complexx:
            return self.get_complex(vary)
        else:
            return self.get_simple(vary)

    def get_sample_length(self, vary):
        return self.len + \
               vary * (np.random.randint(self.len // 2) - self.len // 4)

    def get_complex(self, vary):
        length = self.get_sample_length(vary)
        ret_x = np.zeros((self.nDims, length), dtype=int)
        ret_y = np.zeros((self.nDims, length), dtype=int) - 1

        for char in range(self.nChars):
            ix = np.random.randint(2 * self.nChars) + self.buffer
            while ix < length - self.buffer - self.char_lens[char]:
                ret_x[char, ix:ix + self.char_lens[char]] = 1
                ret_y[char, ix] = char
                ix += self.char_lens[char] + \
                      np.random.exponential(self.nChars**2.) + 1

        ret_y2 = [char for column in ret_y.T for char in column if char > -1]

        return ret_x, ret_y2

    def get_simple(self, vary):
        """
        simple - Implies a character is written only after the previous
                 one is done printing.
        :param vary: Make the slab of variable length
        :return: The Scribe
        """
        length = self.get_sample_length(vary)
        ret_x = np.zeros((self.nDims, length), dtype=int)
        ret_y = []

        ix = np.random.exponential(self.buffer) + self.buffer
        while ix < length - self.buffer - self.nChars:
            char = np.random.randint(self.nChars)
            ret_x[char, ix:ix + self.char_lens[char]] = 1
            ret_y += [char]
            ix += self.char_lens[char] + \
                  np.random.exponential(self.buffer // 2) + 1

        return ret_x, ret_y


if __name__ == "__main__":
    import pickle
    import sys
    from utils import slab_print

    if len(sys.argv) < 2:
        print('Usage \n'
              '{} <out_file_name> [num_chars=4] [avg_sequence_len=30] '
              '[complex=True] [variable_length=True]'.format(sys.argv[0]))
        sys.exit()

    out_file_name = sys.argv[1]
    out_file_name += '.pkl' if not out_file_name.endswith('.pkl') else ''

    try:
        nChars = int(sys.argv[2])
    except IndexError:
        nChars = 4

    try:
        avg_seq_len = int(sys.argv[3])
    except IndexError:
        avg_seq_len = 30

    try:
        complx = sys.argv[4].lower() in ("yes", "true", "t", "1")
    except IndexError:
        complx = True

    try:
        variable_len = sys.argv[5].lower() in ("yes", "true", "t", "1")
    except IndexError:
        variable_len = True

    scribe = RowScribe(nChars, avg_seq_len, buffer_len=avg_seq_len // 10)

    xs = []
    ys = []
    for i in range(1000):
        x, y = scribe.get_data(complx, variable_len)
        xs.append(x)
        ys.append(y)
        print(y)
        slab_print(x)

    print('Output: {}\n'
          'Char set size: {}\n'
          '(Avg.) Len: {}\n'
          'Varying Length: {}\n'
          'Complex Scribe: {}\n'.format(
        out_file_name, nChars, avg_seq_len, variable_len, complx, ))

    chars = [str(x) for x in range(nChars)]
    with open(out_file_name, 'wb') as f:
        pickle.dump({'x': xs, 'y': ys, 'chars': chars}, f, -1)