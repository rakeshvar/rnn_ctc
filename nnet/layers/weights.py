import numpy as np
import theano as th


def orthonormal_wts(n, m):
    nm = max(n, m)
    svd_u = np.linalg.svd(np.random.randn(nm, nm))[0]
    return svd_u.astype(th.config.floatX)[:n, :m]


def stacked_ortho_wts(n, m, copies, name=None):
    return share(
        np.hstack([orthonormal_wts(n, m) for _ in range(copies)]),
        name=name)


def init_wts(*argv):
    return 1 * (np.random.rand(*argv) - 0.5)


def share(array, dtype=th.config.floatX, name=None):
    return th.shared(value=np.asarray(array, dtype=dtype), name=name)