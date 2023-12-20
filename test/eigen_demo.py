import numpy as np

# 求Laplacian Matrix中特征值与特征向量
I = np.identity(6)
A = np.array([[0, 1, 0, 0, 1, 0],
              [1, 0, 1, 0, 1, 0],
              [0, 1, 0, 1, 0, 0],
              [0, 0, 1, 0, 1, 1],
              [1, 1, 0, 1, 0, 0],
              [0, 0, 0, 1, 0, 0]])

v = A.sum(axis=0)
D = I
for i in range(I.shape[0]):
    D[i, i] = v[i]
Laplacian_matrix = D - A
eigenvalues, eigenvectors = np.linalg.eig(Laplacian_matrix)
print(eigenvalues)
print(eigenvalues.sum())
print(eigenvectors)

print(Laplacian_matrix @ eigenvectors[:, 0])

print(eigenvectors@np.diag(eigenvalues)@eigenvectors.T)
print(eigenvectors@eigenvectors.T)

# A = np.array([[1, 2], [3, 4]])
# eigenvalues, eigenvectors = np.linalg.eig(A)
# print(eigenvalues)
# print(eigenvectors)
# A_h = eigenvalues[0] * (eigenvectors[:, 0].reshape(-1, 1) @ eigenvectors[:, 0].reshape(1, -1)) + eigenvalues[1] * (
#         eigenvectors[:, 1].reshape(-1, 1) @ eigenvectors[:, 1].reshape(1, -1))
# print(A_h)

A = np.array([[4, 1, 1],
              [1, 2, 1],
              [3, 2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)
print(eigenvectors)

print(np.linalg.norm(eigenvectors[:,0]))

# 向量连续左乘矩阵A,最终会收敛到头附近的特征向量
tmp=A
for i in range(10):
    tmp=tmp@A
v=np.array([0.577,-0.577,-0.577])
b=tmp@v
print(tmp@v/np.linalg.norm(b))

# 对称矩阵
# 正交矩阵
# 特征值(本征值)
# 特征向量(本征向量)
# 所有特征值的全体叫做A的谱
# 特征向量不能由特征值唯一确定,不同特征值对应的特征向量不会相等,一个特征向量只能属于一个特征值
#