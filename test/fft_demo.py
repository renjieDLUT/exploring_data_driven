import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('./res/lena.png',0)
f=np.fft.fft2(img)
print(f)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift)) # 将零频率成分移动到频域图像的中心位置
magnitude_spectrum2 = 20*np.log(np.abs(f)) # 未移动

plt.subplot(221)
plt.imshow(img,cmap='gray')
plt.title("original")
plt.subplot(222)
plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('magnitude_spectrum')

plt.subplot(223)
plt.imshow(magnitude_spectrum2,cmap='gray')
plt.title('magnitude_spectrum2')

#  复数数组
print(fshift,"fshift",type(fshift),fshift.shape,fshift.size)

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()