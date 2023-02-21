import torch.optim as optim
from loss import *
from core.network import *
from utils.tools import *
from utils.data import *
import torch
import time

torch.multiprocessing.set_sharing_strategy('file_system')

config = {
    "alpha": 0.1,
    "optimizer": {
        "type": optim.RMSprop,
        "optim_params": {
            "lr": 1e-5,
            "weight_decay": 10**-5
        },
        "lr_type": "step"
    },
    "info": "[Net]",
    "step_continuation": 20,
    "resize_size": 256,
    "crop_size": 224,
    "batch_size": 64,
    "net": AlexNet,
    # 数据
    "dataset": "cifar10",
    "topK": -1,
    "n_class": 10,  # 类被
    # 训练次数
    "epoch": 150,
    "test_map": 15,
    "save_path": "./save/Net",
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "bit": [48],
}


# 使用测试数据集测试模型并打印测试图像精度的函数
def testAccuracy(model, test_loader):

    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for images, labels, ind in test_loader:
            # run the model on the test set to predict labels
            hash, outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            _, label = torch.max(labels, 1)
            accuracy += (predicted == label).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return (accuracy)


def train_val(config, bit):
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = cifar_dataset(
        config)
    device = config["device"]
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](
        net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = ContrasiveLoss(config, bit)
    criterion_1 = CrossEntropyLoss()

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." %
              (config["info"], epoch + 1, config["epoch"], current_time, bit,
               config["dataset"]),
              end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u, y = net(image)

            loss = criterion(u, label.float(), ind, config)
            loss2 = criterion_1(y, label.float())
            sumloss = loss + loss2
            train_loss += sumloss

            sumloss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        accuracy = testAccuracy(net, test_loader)
        print("\b\b\b\b\b\b\b loss:%.3f accuracy:%.2f" %
              (train_loss, accuracy))
        # print("\b\b\b\b\b\b\b loss:%.3f accuracy" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader,
                                net, bit, epoch, num_dataset)


if __name__ == "__main__":
    config["pr_curve_path"] = f"./log/alexnet/Net_{config['dataset']}_48.json"
    train_val(config, 48)
