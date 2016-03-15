import sys
import pickle
import utils
from scribe.scribe import Scribe

if len(sys.argv) < 2:
    print('Usage:'
          '\n python3 {} <output_file_name> [configurations]'
          'Generates data based on the configuration files.'.format(sys.argv[0]))
    sys.exit(-1)

out_file_name = sys.argv[1]
if not out_file_name.endswith('.pkl'):
    out_file_name += '.pkl'

args = utils.read_args(sys.argv[2:])
scriber = Scribe(**args['scribe_args'])
alphabet_chars = scriber.alphabet.chars

xs = []
ys = []
for i in range(args['num_samples']):
    x, y = scriber.get_sample()
    xs.append(x)
    ys.append(y)
    print(y, ''.join(alphabet_chars[i] for i in y))
    utils.slab_print(x)

with open(out_file_name, 'wb') as f:
    pickle.dump({'x': xs, 'y': ys, 'chars': alphabet_chars}, f, -1)

print(scriber)
print('Generated dataset:', out_file_name)
