import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, make_interp_spline

# 创建示例数据点
x = []
y = []
z = []
a_0, a_1, a_2, a_3 = 1.0, 0.1, 0.005, 0.001

for i in np.linspace(0., 100., 1000):
    x.append(i)
    tmp = a_0 + a_1 * i + a_2 * pow(i, 2) + a_3 * pow(i, 3)
    y.append(tmp)
    if len(x) == 1:
        z.append(0)
    else:
        z.append(math.sqrt(pow(x[-1] - x[-2], 2) + pow(y[-1] - y[-2], 2)))
x = np.asarray(x)
y = np.asarray(y)
z = np.asarray(z)
z = np.cumsum(z)
print(z)

# 拟合
x_fit = np.polyfit(z, x, deg=4)
y_fit = np.polyfit(z, y, deg=4)
print(x_fit)
print(y_fit)

#
polynomial_x = np.poly1d(x_fit)
polynomial_y = np.poly1d(y_fit)

res_x = []
res_y = []
for i in np.linspace(0., z[-1], len(z)):
    res_x.append(polynomial_x(i))
    res_y.append(polynomial_y(i))
plt.plot(x, y, color='r')
plt.plot(res_x, res_y, color="b")

plt.show()
