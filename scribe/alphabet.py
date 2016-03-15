import numpy as np

class Alphabet():
    def __init__(self, bitmaps, chars=None):
        self.bitmaps = bitmaps
        self.n = len(bitmaps)
        self.chars = chars if chars else [str(i) for i in range(len(bitmaps))]

        self.maxHt = max([bitmap.shape[0] for bitmap in bitmaps])
        self.maxWd = max([bitmap.shape[1] for bitmap in bitmaps])

    def get_char(self, index=None):
        if index is None:
            index = np.random.choice(self.n)

        bitmap = self.bitmaps[index]
        char = self.chars[index]
        return index, char, bitmap

    def __str__(self):
        ret = ''
        for c, b in zip(self.chars, self.bitmaps):
            slab = '\n'.join((''.join('# '[p] for p in r) for r in b))
            ret += '\n{}:\n{}'.format(c, slab)
        return ret