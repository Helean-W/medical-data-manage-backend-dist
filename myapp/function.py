import os
import re
import cv2
import uuid
import base64
import numpy as np
import shutil
import zipfile
import pydicom
import SimpleITK
from models.grading import transformer
from .models import *

import torch
import torch.nn as nn
from torchvision import transforms as T
from scipy import misc
from models.segment.Code.model_lung_infection.InfNet_Res2Net import Inf_Net as Network
from models.segment.Code.utils.dataloader_LungInf import test_dataset
from models.grading.resnet import resnet50
from PIL import Image

from IPython import embed


def unzip(fName):
    zfile = zipfile.ZipFile(os.path.join("./", "resources", "zipTemp", fName))
    zfile.extractall(os.path.join("./", "resources", "dcmTemp"))
    zfile.close()


def importZip(Position):
    rootdir = "./resources/dcmTemp"
    url = "http://122.144.180.37:8001/"
    list = os.listdir(rootdir)
    for item in list:
        name = (
            str(uuid.uuid4()).replace("-", "") + "." + item.split(".")[1]
        )  # 对每个dcm文件产生唯一id
        path = os.path.join(rootdir, item)
        info = {}
        info["PatientSex"] = ""
        info["PatientAge"] = ""
        if Position == "胰腺":
            dcm = pydicom.dcmread(path, force=True)  # 读取dicom文件
            # info["PatientName"] = str(dcm.PatientName).split(" ")[0]  #清洗姓名
            info["PatientAge"] = dcm.PatientAge
            info["PatientAge"] = re.sub("\D", "", info["PatientAge"])  # 把字符串中的数字提取出来
            info["PatientAge"] = int(info["PatientAge"])  # 把字符串的064转化成数字64
            if dcm.PatientSex == "F":
                info["PatientSex"] = "女"
            elif dcm.PatientSex == "M":
                info["PatientSex"] = "男"
            else:
                info["PatientSex"] = "不详"
            # info['PatientSex'] = dcm.PatientSex
        info["Position"] = Position
        info["Url"] = url + "resources/" + name
        # 存入数据库
        Patient.objects.create(
            gender=info["PatientSex"],
            age=info["PatientAge"],
            position=info["Position"],
            url=info["Url"],
        )
        # 重命名后移入resources目录下长期存储
        os.rename(path, name)
        shutil.move(name, "./resources")


def importSingle(info, path, fname):
    url = "http://122.144.180.37:8001/"
    name = (
        str(uuid.uuid4()).replace("-", "") + "." + fname.split(".")[1]
    )  # 对每个dcm文件产生唯一id
    print(name)
    if info["position"] == "胰腺":
        dcm = pydicom.dcmread(path, force=True)  # 读取dicom文件
        # if(dcm.PatientName != ''):
        #     info["name"] = str(dcm.PatientName).split(" ")[0]
        if info["gender"] == "" and dcm.PatientSex != "":
            if dcm.PatientSex == "F":
                info["gender"] = "女"
            elif dcm.PatientSex == "M":
                info["gender"] = "男"
        if info["age"] == "" and dcm.PatientAge != "":
            info["age"] = dcm.PatientAge
            info["age"] = re.sub("\D", "", info["age"])
            info["age"] = int(info["age"])
    info["url"] = url + "resources/" + name
    os.rename(path, name)
    Patient.objects.create(
        gender=info["gender"],
        age=info["age"],
        position=info["position"],
        url=info["url"],
    )
    shutil.move(name, "./resources")


def deleteDcm(url):
    dcm_file_path = url.split("/", 3)[3]  #  resources/xxxxx.dcm
    dcm_file_path = os.path.join("./", dcm_file_path)
    os.remove(dcm_file_path)


def get_pixels_by_simpleitk(dicom_dir):
    ds = SimpleITK.ReadImage(dicom_dir)
    img_array = SimpleITK.GetArrayFromImage(ds)
    img_array[img_array == -2000] = 0
    return img_array


def dcm2img(dcm_url):
    dcm_file_path = dcm_url.split("/", 3)[3]  #  resources/xxxxx.dcm
    dcm_file_path = os.path.join("./", dcm_file_path)
    jpgName = "temp.jpg"
    jpg_file_path = os.path.join("./", "resources", "jpgTemp", jpgName)
    img = get_pixels_by_simpleitk(dcm_file_path)
    # print(np.min(img), np.max(img)-np.min(img))
    scaled_img = cv2.convertScaleAbs(
        img - np.min(img), alpha=(255.0 / min(np.max(img) - np.min(img), 10000))
    )
    cv2.imwrite(jpg_file_path, scaled_img[0])
    with open(jpg_file_path, "rb") as f:
        image = f.read()
        image_base64 = base64.b64encode(image)
        return image_base64


def img2base64(img_url):
    img_path = img_url.split("/", 3)[3]  # resources/xxxxx.dcm
    img_path = os.path.join("./", img_path)
    with open(img_path, "rb") as f:
        image = f.read()
        image_base64 = base64.b64encode(image)
        return image_base64


def show_label(label, name):
    img = misc.toimage(label, cmin=0.0, cmax=1.0)
    img = img.convert("RGBA")
    x, y = img.size
    for i in range(x):
        for j in range(y):
            color = img.getpixel((i, j))
            Mean = np.mean(list(color[:-1]))
            if Mean < 100:
                color = (255, 97, 0, 0)
            # elif Mean < 200:
            #     color = (255, 97, 0, 200)
            else:
                color = (25, 25, 112, 150)
            img.putpixel((i, j), color)

    img.save(name)
    return img


def seg_model(image_path, processed_path, mask_path):
    print("----run the segmentation model")

    pth_path = "models/segment/Snapshots/save_weights/Inf-Net/Inf-Net-100.pth"
    testsize = 352

    # 分割结果存储路径
    # label_path = "models/segment/Lung infection segmentation/Inf-Net/Labels/"
    # label_path = './resources/diagnosis/temp/label'
    # mask_path = "models/segment/Results/Lung infection segmentation/Inf-Net/Mask/"
    # mask_path = './resources/diagnosis/temp/mask'

    device = torch.device("cpu")
    model = Network()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = test_dataset(image_path, testsize)

    image, name = test_loader.load_data()

    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(
        image
    )

    # embed()

    res = lateral_map_2
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    name = "temp.png"

    # embed()

    # misc.imsave(os.path.join(label_path, name), res)
    # Image.fromarray(res).convert('RGB').save(save_path + name)

    icon = show_label(res, os.path.join(mask_path, name))
    img = Image.open(image_path)
    img = img.convert("RGBA")
    img_w, img_h = img.size
    icon = icon.resize((img_w, img_h), Image.ANTIALIAS)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    layer.paste(icon, (0, 0))
    out = Image.composite(layer, img, layer)
    out.save(processed_path)

    print("Segmentation Finished!")
    with open(processed_path, "rb") as f:
        image = f.read()
        image_base64 = base64.b64encode(image)
        return image_base64


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
    pth_path = "models/grading/12.pkl"

    net = resnet50(pretrained=False)
    net.fc = nn.Linear(2048, 5)
    device = torch.device("cpu")
    model_dict = torch.load(pth_path, map_location=device)
    net.load_state_dict(
        {k.replace("module.", ""): v for k, v in model_dict["net"].items()}
    )

    net.to(device)
    net.eval()

    image = cv2.imread(image_path)
    image = augment(image)
    image = torch.tensor(image)
    image = image.unsqueeze(dim=0)

    _, y_pred = net(image)

    # y_pred, y_pred1 = net(x)
    prediction = y_pred.max(1)[1]
    result = str(prediction.numpy()[0])
    return result
