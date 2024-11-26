# -*- coding: utf-8 -*-
# @Time    : 2024/9/19 15:07
# @Author  : BitYang
# @FileName: net.py
# @Software: PyCharm

from efficient_kan import KAN
import torch.nn as nn
import timm

class MobileNetKAN(nn.Module):
    def __init__(self, ):
        super(MobileNetKAN, self).__init__()
        self.base = timm.create_model('mobilenetv2_100', pretrained=False)
        self.neck = KAN([1000, 128])
        self.head = KAN([128, 1])

    def forward(self, x):
        x = self.base(x)
        x = self.neck(x)
        x = self.head(x)
        return x