from torchvision import transforms
from torchvision import models
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from flask import Flask, render_template, request
import json
import base64
import warnings
import torch

warnings.filterwarnings("ignore")


class AlexNet(nn.Module):

    def __init__(self, hash_bit, n_class, weights=models.AlexNet_Weights.DEFAULT):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(weights=weights)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

        self.classify_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hash_bit, n_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        y = self.classify_layer(x)
        return x, y


# 能用就行~
device = torch.device('cpu')

img_dir = "/dataset/imagenet/"
with open("data/imagenet/database.txt", "r") as f:
    trn_img_path = np.array([img_dir + item.split(" ")[0] for item in f.readlines()])
with open("data/imagenet/class.txt", "r", encoding='utf-8') as f:
    classes = np.array([item.strip('\n').split("\t") for item in f.readlines()])
save_path = "save/imagenet_64bits_4/"
trn_binary = np.load(save_path + "trn_binary.npy")

# # 加载模型
print("加载模型中。。。。。。。")
# 这里写模型路径
model_name = 'model.pt'
model_state_dict = torch.load(save_path + model_name, map_location=device)
# 哈希码长度64
model = AlexNet(64, 100)
model.load_state_dict(model_state_dict)
model.eval()
print("模型加载成功")

# transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform = transforms.Compose([transforms.Resize(256)] + [transforms.CenterCrop(224)] + [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# 输入路径，返回哈希码
def detect(source):
    img = Image.open(source).convert('RGB')
    img = transform(img).unsqueeze(0)
    res = model(img)
    qB = res[0].sign()[0].detach().numpy()
    cls = torch.max(res[1], 1)[1].item()
    return qB, cls


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def retrival(qB, start=0, end=50):
    # 通过哈希码计算汉明距离
    hamm = CalcHammingDist(qB, trn_binary)
    # 计算最近的n个距离的索引
    ind = np.argsort(hamm)[start:end]
    # 返回结果的真值
    # 返回结果的汉明距离
    result_hamm = hamm[ind].astype(int)
    result_path = trn_img_path[ind]
    result_code = trn_binary[ind]
    result = []
    for hmm, path, code in zip(result_hamm, result_path, result_code):
        row = {}
        row["hmm"] = int(hmm)
        with open(path, 'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
        row["img"] = img_stream
        row["code"] = convert0(code)
        result.append(row)
    return result


# 将+1，-1 -> 01串
def convert0(code):
    return "".join(code.astype(int).astype(str).tolist()).replace("-1", "0")


def convert1(code):
    code = list(code)
    code = [-1.0 if (c == "0") else 1.0 for c in code]
    return np.array(code)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    f = request.files['file']
    qB, cls = detect(f)
    qB_binary = convert0(qB)
    # print(qB_binary)
    result = retrival(qB, end=50)
    response = {"qB": qB_binary, "class": classes[cls][0], "result": result}
    # print(response)
    return json.dumps(response, ensure_ascii=False)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
