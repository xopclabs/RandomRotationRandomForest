import numpy as np


def random_rotation_matrix(n, random_state=None):
    random = np.random.default_rng(random_state)
    r = random.normal(size=(n, n))
    Q, R = np.linalg.qr(r)
    M = np.dot(Q, np.diag(np.sign(np.diag(R))))
    if np.linalg.det(M) < 0:
        M[:, 0] = -M[:, 0]
    return M.astype(np.float32)
