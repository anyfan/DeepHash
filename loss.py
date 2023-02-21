import torch.nn as nn
from utils.tools import *



class ContrasiveLoss(torch.nn.Module):

    def __init__(self, config, bit):
        super(ContrasiveLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"],
                             bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"],
                             config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        # 欧氏距离 ^ 2
        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        # y=0,有公共的标签就算做相似(计算内积再判断是否大于0，就能判断两张图片是否是相似的，从而得到y值)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (1 - u.abs()).abs().mean()

        return loss1 + loss2


class CrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, u, y):
        return self.criterion(u, y)


