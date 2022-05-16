import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class FM(Module):
    def __init__(self, config, p):
        super(FM, self).__init__()
        # p : 特征的个数
        self.p = p
        # k : 隐藏层的维度
        self.k = config["latent_dim"]
        # FM的线性部分，即 ∑WiXi
        self.linner = nn.Linear(self.p, 1, bias=True)
        # 隐向量的维度维n×k
        self.v = Parameter(torch.randn(self.p, self.k), requires_grad=True)

    def forward(self, x):
        # 1. 线性部分
        linner_part = self.linner(x)
        # 2. 矩阵相乘 (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v)
        # 3. 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))

        output = linner_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        return output
