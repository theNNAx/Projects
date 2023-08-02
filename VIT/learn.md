### 上采样

```python

import cv2
img = cv2.imread('a.jpg')

# 定义升采样倍数
scale_percent = 150  # 倍数为200%

# 计算升采样后的尺寸
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# 使用最近邻插值法进行升采样
resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)

# 显示原图和升采样后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', resized_image)

```

[SIFT - Scale-Invariant Feature Transform (weitz.de)](http://weitz.de/sift/index.html?size=large)

[高斯拉普拉斯算子（Laplacian of Gaussian, LoG） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/92143464)


$$
G(x,y)=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}
$$

$$
\nabla^2=\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}
$$


$$
h(x,y)=-\frac{1}{\pi\sigma^4}\left[1-\frac{x^2+y^2}{2\sigma^2}\right]e^{-\frac{x^2+y^2}{2\sigma^2}}
$$

$$

$$
