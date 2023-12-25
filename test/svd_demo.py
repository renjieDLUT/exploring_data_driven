import copy

import numpy as np

np.set_printoptions(precision=3)


def svd(A: np.ndarray):
    AAt = A @ A.T
    AtA = A.T @ A

    eig_value, eig_vec = np.linalg.eig(AAt)
    print(eig_value, eig_vec, sep="\n")
    U = copy.deepcopy(eig_vec)
    ind = np.argsort(eig_value)[::-1]

    for i in range(len(ind)):
        U[:, i] = eig_vec[:, ind[i]]

    eig_value, eig_vec = np.linalg.eig(AtA)
    print(eig_value, eig_vec, sep="\n")
    V = copy.deepcopy(eig_vec)
    ind = np.argsort(eig_value)[::-1]

    diag = np.zeros_like(A, dtype=np.float32)
    for i in range(len(ind)):
        V[:, i] = eig_vec[:, ind[i]]
        if i < A.shape[0]:
            diag[i, i] = np.sqrt(eig_value[ind[i]])

    return U, diag, V


A = np.array([[1., 2., 0.],
              [2., 3., 4.]])

U, diag, V = svd(A)
print(f'U:{U}')
print(f'diag:{diag}')
print(f'V:{V}')
print(U @ diag @ V.T)

# PCA 降维
from PIL import Image
from matplotlib import pyplot as plt

fig: plt.Figure = plt.figure()
raw_img: Image.Image = Image.open("./res/lena.png")
img = raw_img.convert(mode='L')
axs = []
for i in range(9):
    axs.append(fig.add_subplot(3, 3, i + 1))
axs[0].imshow(img, label='raw')
img = np.asarray(img)

U, S, Vt = np.linalg.svd(img)
K = [1, 10, 20, 50, 100, 200, 400, 500]
for i in range(len(K)):
    k = K[i]
    u = U[:, :k]
    s = S[:k]
    vt = Vt[:k, :]

    diag = np.diag(s)
    compressed = u @ diag @ vt
    axs[i + 1].imshow(compressed)
    axs[i + 1].set_title(f'k={k}')
    print(f'{k}:{s[k - 1]}')
plt.show()
