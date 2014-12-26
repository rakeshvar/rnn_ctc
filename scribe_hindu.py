#!/usr/bin/env python3
"""
'Scribe' a random numerical 'text' on to a 'slab'.
  'text' - A sequence based on an alphabet [0, 1, 2 ...n_chars)
  'slab' - An numpy matrix

  Example:- The text [1, 0, 3, 5, 4, 2] can be written as
     0¦                                 ¦
     1¦             ██  ███       ██    ¦
     2¦            █  █ █        █  █   ¦
     3¦              █  ███   █  █ █    ¦
     4¦   █    ██  █  █   █  █  █ █     ¦
     5¦   █   █  █  ██  █ █ ████ ████   ¦
     6¦   █   █  █      ███   █         ¦
     7¦        ██           █           ¦

"""
import numpy as np

numbers_ = [
    [
        [0,1,1,0],
        [1,0,0,1],
        [1,0,0,1],
        [0,1,1,0],
    ],
    [
        [1,1,0],
        [0,1,0],
        [0,1,0],
        [1,1,1],
    ],
    [
        [0,1,1,0],
        [1,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,1,1,1],
    ],
    [
        [0,1,1,0],
        [1,0,0,1],
        [0,0,1,0],
        [1,0,0,1],
        [0,1,1,0],
    ],
    [
        [0,0,1,0,0,1],
        [0,1,0,0,1,0],
        [1,1,1,1,0,0],
        [0,0,1,0,0,0],
        [0,1,0,0,0,0],
    ],
    [
        [1,1,1],
        [1,0,0],
        [1,1,1],
        [0,0,1],
        [1,0,1],
        [1,1,1],
    ]
]

numbers = [np.asarray(num) for num in numbers_]
maxHt = max([num.shape[0] for num in numbers])
maxWd = max([num.shape[1] for num in numbers])
nChars = len(numbers)


class NumberScribe():
    def __init__(self, avg_seq_len, noise=0., vbuffer=2, hbuffer=3,):
        self.len = avg_seq_len
        self.hbuffer = hbuffer
        self.vbuffer = vbuffer
        self.nDims = maxHt + vbuffer
        self.noise = noise

    def get_sample_length(self, vary):
        return self.len + \
               vary * (np.random.randint(self.len // 2) - self.len // 4)

    def get_sample(self, vary):
        length = self.get_sample_length(vary)
        ret_x = np.zeros((self.nDims, length), dtype=float)
        ret_y = []

        ix = np.random.exponential(self.hbuffer) + self.hbuffer
        while ix < length - self.hbuffer - maxWd:
            char = np.random.randint(nChars)
            ht, wd = numbers[char].shape
            at_ht = np.random.randint(self.vbuffer + maxHt - ht + 1)
            ret_x[at_ht:at_ht+ht, ix:ix+wd] += numbers[char]
            ret_y += [char]
            ix += wd + np.random.randint(self.hbuffer+1)+1

        ret_x += self.noise * np.random.normal(size=ret_x.shape,)
        ret_x = np.clip(ret_x, 0, 1)
        return ret_x, ret_y


if __name__ == "__main__":
    import pickle
    import sys
    from print_utils import slab_print

    if len(sys.argv) < 2:
        print('Usage \n'
              '{} <out_file_name> [avg_sequence_len=30] [noise=0.0]'
              '[variable_length=True]'.format(sys.argv[0]))
        sys.exit()

    out_file_name = sys.argv[1]
    out_file_name += '.pkl' if not out_file_name.endswith('.pkl') else ''

    try:
        avg_seq_len = int(sys.argv[2])
    except IndexError:
        avg_seq_len = 30

    try:
        noise = float(sys.argv[3])
    except IndexError:
        noise = 0.0

    try:
        variable_len = sys.argv[4].lower() in ("yes", "true", "t", "1")
    except IndexError:
        variable_len = True

    scribe = NumberScribe(avg_seq_len, noise)

    xs = []
    ys = []
    for i in range(1000):
        x, y = scribe.get_sample(variable_len)
        xs.append(x)
        ys.append(y)
        print(y)
        slab_print(x)

    print('Output: {}\n'
          'Char set size: {}\n'
          '(Avg.) Len: {}\n'
          'Varying Length: {}\n'
          'Noise Level: {}'.format(
        out_file_name, nChars, avg_seq_len, variable_len, noise))

    with open(out_file_name, 'wb') as f:
        pickle.dump({'x': xs, 'y': ys, 'nChars': nChars}, f, -1)