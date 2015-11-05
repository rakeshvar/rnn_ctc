import numpy as np
irand = np.random.randint

class Scribe():
    def __init__(self, alphabet, noise, vbuffer, hbuffer,
                 avg_seq_len = None,
                 varying_len = True,
                 nchars=None):
        self.alphabet = alphabet
        self.noise = noise
        self.hbuffer = hbuffer
        self.vbuffer = vbuffer

        self.avg_len = avg_seq_len
        self.varying_len = varying_len
        self.nchars = nchars

        self.nDims = alphabet.maxHt + vbuffer

        self.shuffled_char_indices = np.arange(self.alphabet.n)
        self.shuffled_char_pointer = 0


    def get_sample_of_random_len(self):
        length = int(self.avg_len * np.random.uniform(.75, 1.25))
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
        if self.nchars:
            return self.get_sample_of_n_chars(self.nchars)

        if self.varying_len:
            return self.get_sample_of_random_len()
        else:
            return self.get_sample_of_len(self.avg_len)