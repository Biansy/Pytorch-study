# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import matplotlib.pyplot as plt

#linspace 在-1到1之间，均匀生成100个数
#unsqueeze 把一维数据变为二维数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)


# 画图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
