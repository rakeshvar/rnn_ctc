import numpy as np
irand = np.random.randint

class Scribe():
    def __init__(self, alphabet, noise, vbuffer, hbuffer,
                 avg_seq_len = None,
                 varying_len = True,
                 nchars_per_sample=None):
        self.alphabet = alphabet
        self.noise = noise
        self.hbuffer = hbuffer
        self.vbuffer = vbuffer

        self.avg_len = avg_seq_len
        self.varying_len = varying_len
        self.nchars_per_sample = nchars_per_sample

        self.nDims = alphabet.maxHt + vbuffer

        self.shuffled_char_indices = np.arange(self.alphabet.n)
        self.shuffled_char_pointer = 0

        self.nClasses = len(alphabet.chars)
        self.len_range = (3*self.avg_len//4,  5*self.avg_len//4)

    def get_sample_of_random_len(self):
        length = np.random.randint(*self.len_range)
        return self.get_sample_of_len(length)

    def get_next_char(self):
        if self.shuffled_char_pointer == len(self.shuffled_char_indices):
            np.random.shuffle(self.shuffled_char_indices)
            self.shuffled_char_pointer = 0

        self.shuffled_char_pointer += 1
        return self.alphabet.get_char(self.shuffled_char_indices[
            self.shuffled_char_pointer-1])

    def get_sample_of_len(self, length):
        image = np.zeros((self.nDims, length), dtype=float)
        labels = []

        at_wd = np.random.exponential(self.hbuffer) + self.hbuffer
        while at_wd < length - self.hbuffer - self.alphabet.maxWd:
            index, char, bitmap = self.get_next_char()
            ht, wd = bitmap.shape
            at_ht = irand(self.vbuffer + self.alphabet.maxHt - ht + 1)
            image[at_ht:at_ht+ht, at_wd:at_wd+wd] += bitmap
            at_wd += wd + irand(self.hbuffer)
            labels.append(index)

        image += self.noise * np.random.normal(size=image.shape,)
        image = np.clip(image, 0, 1)
        return image, labels

    def get_sample_of_n_chars(self, n):
        gaps = irand(self.hbuffer, size=n+1)
        labels_bitmaps = [self.get_next_char() for _ in range(n)]
        labels, _, bitmaps = zip(*labels_bitmaps)
        length = sum(gaps) + sum(b.shape[1] for b in bitmaps)
        image = np.zeros((self.nDims, length), dtype=float)

        at_wd = gaps[-1]
        for i, bitmap in enumerate(bitmaps):
            ht, wd = bitmap.shape
            at_ht = irand(self.vbuffer + self.alphabet.maxHt - ht + 1)
            image[at_ht:at_ht+ht, at_wd:at_wd+wd] += bitmap
            at_wd += wd + gaps[i]

        image += self.noise * np.random.normal(size=image.shape,)
        image = np.clip(image, 0, 1)
        return image, labels

    def get_sample(self):
        if self.nchars_per_sample:
            return self.get_sample_of_n_chars(self.nchars_per_sample)

        if self.varying_len:
            return self.get_sample_of_random_len()
        else:
            return self.get_sample_of_len(self.avg_len)

    def __repr__(self):
        if self.nchars_per_sample:
            cps = self.nchars_per_sample
            len = 'Varies (to fit the {} of chars per sample)'.format(cps)
        else:
            cps = 'Depends on the random length'
            len = 'Avg:{} Range:{}'.format(self.avg_len, self.len_range)

        ret = ('Scribe:'
               '\n  Alphabet: {}'
               '\n  Noise: {}'
               '\n  Buffers (vert, horz): {}, {}'
               '\n  Characters per sample: {}'
               '\n  Length: {}'
               '\n  Height: {}'
               '\n'.format(
            ''.join(self.alphabet.chars),
            self.noise,
            self.hbuffer,
            self.vbuffer,
            cps, len,
            self.nDims,
            ))
        return ret