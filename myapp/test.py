import cv2
import numpy as np
import sys
import pydicom
import SimpleITK
import torchvision
import cv2
import uuid

sys.path.append("../models/grading")
import transformer as transformer
from resnet import resnet50

import torch
import torch.nn as nn
from torchvision import transforms as T
from scipy import misc

from PIL import Image

from IPython import embed


def augment(data):
    crop_h = data.shape[0]

    cp = transformer.CenterCrop_np(crop_h)
    rs = transformer.Scale(512, 512)

    nor = T.Normalize(mean=[0.218, 0.306, 0.442], std=[0.100, 0.142, 0.203])

    tf = T.Compose([T.ToTensor(), nor])

    data = tf(rs(cp(data)))

    return data


def grading_model(image_path):
    print("----run the grading model")
    pth_path = "/home/ubuntu/medical-data-manage-backend/models/grading/12.pkl"

    net = resnet50(pretrained=False)
    net.fc = nn.Linear(2048, 5)
    device = torch.device("cpu")
    model_dict = torch.load(pth_path, map_location=torch.device("cpu"))
    net.load_state_dict(
        {k.replace("module.", ""): v for k, v in model_dict["net"].items()}
    )
    net.to(device)
    net.eval()
    image = cv2.imread(image_path)
    image = augment(image)
    image = torch.tensor(image)
    image = image.unsqueeze(dim=0)
    # embed()
    _, y_pred = net(image)

    # y_pred, y_pred1 = net(x)
    prediction = y_pred.max(1)[1]
    print(prediction)
    return prediction


def importSingle(info, path, fname):
    url = "http://122.144.180.37:8001/"
    name = (
        str(uuid.uuid4()).replace("-", "") + "." + fname.split(".")[1]
    )  # 对每个dcm文件产生唯一id
    print(name)
    if info["position"] == "胰腺":
        dcm = pydicom.dcmread(path, force=True)  # 读取dicom文件
        print(dcm.PatientSex == "")
        print(dcm.get("PatientAge"))


if __name__ == "__main__":
    info = {}
    info["position"] = "胰腺"
    path = "/home/ubuntu/medical-data-manage-backend/resources/4562012c2b7e42ec8d35ee414052b155.dcm"
    name = "4562012c2b7e42ec8d35ee414052b155.dcm"
    importSingle(info, path, name)
