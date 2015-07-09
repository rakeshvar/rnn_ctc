import pickle
import numpy as np
import sys
from print_utils import slab_print
from alphabets import hindu_alphabet, ascii_alphabet


class Scribe():
    def __init__(self, alphabet, avg_seq_len, noise=0., vbuffer=2, hbuffer=3,):
        self.alphabet = alphabet
        self.len = avg_seq_len
        self.hbuffer = hbuffer
        self.vbuffer = vbuffer
        self.nDims = alphabet.maxHt + vbuffer
        self.noise = noise

    def get_sample_length(self, vary):
        return self.len + vary * (np.random.randint(self.len // 2)
                                  - self.len // 4)

    def get_sample(self, vary):
        length = self.get_sample_length(vary)
        ret_x = np.zeros((self.nDims, length), dtype=float)
        ret_y = []

        ix = np.random.exponential(self.hbuffer) + self.hbuffer
        while ix < length - self.hbuffer - self.alphabet.maxWd:
            index, char, bitmap = self.alphabet.random_char()
            ht, wd = bitmap.shape
            at_ht = np.random.randint(self.vbuffer +
                                      self.alphabet.maxHt - ht + 1)
            ret_x[at_ht:at_ht+ht, ix:ix+wd] += bitmap
            ret_y += [index]
            ix += wd + np.random.randint(self.hbuffer+1)

        ret_x += self.noise * np.random.normal(size=ret_x.shape,)
        ret_x = np.clip(ret_x, 0, 1)
        return ret_x, ret_y


def main():
    alphabet_name = "ascii"
    avg_seq_len = 30
    noise = 0.05
    variable_len = True

    if len(sys.argv) < 2:
        print('Usage \n'
              '{} <out_file_name> [alphabet={}] [avg_sequence_len={}] '
              '[noise={}] [variable_length={}]'.format(
            sys.argv[0], alphabet_name, avg_seq_len, noise, variable_len))
        sys.exit()

    out_file_name = sys.argv[1]
    out_file_name += '.pkl' if not out_file_name.endswith('.pkl') else ''


    if len(sys.argv) > 2:
        alphabet_name = sys.argv[2]

    if len(sys.argv) > 3:
        avg_seq_len = int(sys.argv[3])

    if len(sys.argv) > 4:
        noise = float(sys.argv[4])

    if len(sys.argv) > 5:
        variable_len = sys.argv[5].lower()[0] in "yt1"

    if alphabet_name == "ascii":
        alphabet = ascii_alphabet
    else:
        alphabet = hindu_alphabet

    print(alphabet)
    scribe = Scribe(alphabet, avg_seq_len, noise)

    xs = []
    ys = []
    for i in range(1000):
        x, y = scribe.get_sample(variable_len)
        xs.append(x)
        ys.append(y)
        print(y, "".join(alphabet.chars[i] for i in y))
        slab_print(x)

    print('Output: {}\n'
          'Char set : {}\n'
          '(Avg.) Len: {}\n'
          'Varying Length: {}\n'
          'Noise Level: {}'.format(
        out_file_name, alphabet.chars, avg_seq_len, variable_len, noise))

    with open(out_file_name, 'wb') as f:
        pickle.dump({'x': xs, 'y': ys, 'chars': alphabet.chars}, f, -1)

if __name__ == '__main__':
    main()