from utils.tools import *
# from core.network import *

import os
import torch
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

# HashNet(ICCV2017)
# paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)
# code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)


class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.scale = 1

    def forward(self, u, y, ind, config):
        u = torch.tanh(self.scale * u)
        S = (y @ y.t() > 0).float()
        sigmoid_alpha = config["alpha"]
        dot_product = sigmoid_alpha * u @ u.t()
        mask_positive = S > 0
        mask_negative = (1 - S).bool()

        neg_log_probe = dot_product + \
            torch.log(1 + torch.exp(-dot_product)) - S * dot_product
        S1 = torch.sum(mask_positive.float())
        S0 = torch.sum(mask_negative.float())
        S = S0 + S1

        neg_log_probe[mask_positive] = neg_log_probe[mask_positive] * S / S1
        neg_log_probe[mask_negative] = neg_log_probe[mask_negative] * S / S0

        loss = torch.sum(neg_log_probe) / S
        return loss


def train_val(config, bit):
    # 训练设备 cpu or cuda
    device = config["device"]
    net = config["net"](bit).to(device)

    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(
        config)

    config["num_train"] = num_train

    optimizer = config["optimizer"]["type"](
        net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = HashNetLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):
        criterion.scale = (epoch // config["step_continuation"] + 1) ** 0.5

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, scale:%.3f, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"], criterion.scale), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:

            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader,
                                dataset_loader, net, bit, epoch, num_dataset)
