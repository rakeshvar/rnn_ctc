import argparse
import pickle
from print_utils import slab_print
from alphabets import hindu_alphabet, ascii_alphabet


################################ Parse Arguments ###########################
from scribe import Scribe


class Formatter(argparse.RawDescriptionHelpFormatter,
                argparse.ArgumentDefaultsHelpFormatter):
    pass

desc = '''Generate Sequence Data.
    Will output a pkl file, with the sequence of variable length matrices x_i
    and their corresponding variable length labels y_i. The pkl file will also
    contain the character set corresponding to the labels.
    Examples:
         python3 {0} data -ly 5 -a hindu # For random five-digit numbers
         python3 {0} data -lx 60
         python3 {0} data -lx 30 -fixed -noise .1
         python3 {0} data -ly 1 -n 8400
         python3 {0} data -ly 5
         python3 {0} data -hbuf 8 -vbuf 8'''.format(
    __file__)

prsr = argparse.ArgumentParser(description=desc, formatter_class=Formatter)

prsr.add_argument('-a', action='store', dest='alphabet',
                  default='ascii',
                  help='The alphabel to be used: hindu, ascii, etc.')
prsr.add_argument('-noise', action='store', dest='noise', type=float,
                  default=.05, help='Nosie')
prsr.add_argument('-n', action='store', dest='nsamples', type=int,
                  default=1000, help='Number of samples')

prsr.add_argument('-lx', action='store', dest='avg_seq_len', type=int,
                  default='30',
                  help='Average length of each image.')
prsr.add_argument('-fixed', dest='varying_len', action='store_false',
                  help='Set the length of each image to be variable.')

prsr.add_argument('-ly', action='store', dest='nchars', type=int,
                  default=0,
                  help='Fixed length of each label sequence. Overrides lx and varx')

prsr.add_argument('-vbuf', action='store', dest='vbuffer', type=int,
                  default=3, help='Vertical buffer')
prsr.add_argument('-hbuf', action='store', dest='hbuffer', type=int,
                  default=3, help='Horizontal buffer')

prsr.add_argument('output_name', action='store',
                  help='Output will be stored to <output_name>.pkl')

prsr.set_defaults(varying_len=True)
args = prsr.parse_args()


###########################################################################

out_file_name = args.output_name
out_file_name += '.pkl' if not out_file_name.endswith('.pkl') else ''

if args.alphabet == "ascii":
    alphabet = ascii_alphabet
else:
    alphabet = hindu_alphabet

print(alphabet)
scribe = Scribe(alphabet=alphabet,
                noise=args.noise,
                vbuffer=args.vbuffer,
                hbuffer=args.hbuffer,
                avg_seq_len=args.avg_seq_len,
                varying_len=args.varying_len,
                nchars=args.nchars)

xs = []
ys = []
for i in range(args.nsamples):
    x, y = scribe.get_sample()
    xs.append(x)
    ys.append(y)
    print(y, "".join(alphabet.chars[i] for i in y))
    slab_print(x)

print('Output: {}\n'
      'Char set : {}\n'.format(out_file_name, alphabet.chars))
for var,val in vars(args).items():
    print("{:12}: {}".format(var, val))

with open(out_file_name, 'wb') as f:
    pickle.dump({'x': xs, 'y': ys, 'chars': alphabet.chars}, f, -1)