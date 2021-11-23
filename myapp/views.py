from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection, transaction
import os
import json
import time
import base64

from .models import *
from .function import *

from IPython import embed

# Create your views here.


def uploadZip(request):
    fp = request.FILES.get("file")
    position = request.POST.get("position")  # 前端传递的当前影像位置
    print("uplaod_file_name => ", fp.name)
    print(position)
    # fp 获取到的上传文件对象
    if fp:
        fpName = str(time.time()) + fp.name
        path = os.path.join("./", "resources", "zipTemp", fpName)
        if fp.multiple_chunks():
            file_yield = fp.chunks()  # 迭代写入文件
            with open(path, "wb") as f:
                for buf in file_yield:  # for情况执行无误才执行 else
                    f.write(buf)
                else:
                    print("大文件上传完毕")
        else:
            with open(path, "wb") as f:
                f.write(fp.read())
                print("小文件上传完毕")

        unzip(fpName)
        importZip(position)  # 传递位置参数，让rename可以向数据库添加记录
        os.remove(path)  # 删除上传的zip文件
    else:
        error = "文件上传为空"
    resp = {"isSuccess": True}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def uploadSingle(request):
    fp = request.FILES.get("file")
    info = {}
    # info["name"] = request.POST.get("name")
    info["gender"] = request.POST.get("gender")
    info["age"] = request.POST.get("age")
    info["position"] = request.POST.get("position")
    print("uplaod_file_name => ", fp.name)
    print(info)

    if fp:
        fpName = fp.name
        path = os.path.join("./", "resources", "dcmTemp", fpName)
        if fp.multiple_chunks():
            file_yield = fp.chunks()  # 迭代写入文件
            with open(path, "wb") as f:
                for buf in file_yield:  # for情况执行无误才执行 else
                    f.write(buf)
                else:
                    print("大文件上传完毕")
        else:
            with open(path, "wb") as f:
                f.write(fp.read())
                print("小文件上传完毕")

        importSingle(info, path, fpName)  # 传入表单信息，文件路径和文件名
    else:
        error = "文件上传为空"
    resp = {"isSuccess": True}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def queryAll(request):
    resp = {"isSuccess": True, "msg": "success"}
    cursor = connection.cursor()
    cursor.execute("""select * from myapp_patient""")
    row = cursor.fetchall()
    resp["ret"] = row
    return HttpResponse(json.dumps(resp), content_type="application/json")


def deleteItem(request):
    msg = "delete success!"
    try:
        del_id = request.GET.get("id")
        del_url = Patient.objects.filter(id=del_id)[0].url
        deleteDcm(del_url)
        Patient.objects.filter(id=del_id).delete()
    except Exception as e:
        msg = "delete failed! " + str(e)
    resp = {"isSuccess": True, "msg": msg}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def viewDcm(request):
    dcm_url = request.GET.get("url")
    img_base64 = dcm2img(dcm_url)
    os.remove("./resources/jpgTemp/temp.jpg")  # 删除临时jpg文件
    return HttpResponse(img_base64, content_type="image/jpeg")


def viewPng(request):
    img_url = request.GET.get("url")
    img_base64 = img2base64(img_url)
    return HttpResponse(img_base64, content_type="image/png")


def viewJpg(request):
    img_url = request.GET.get("url")
    img_base64 = img2base64(img_url)
    return HttpResponse(img_base64, content_type="image/jpeg")


def segImage(request):
    mask_path = "./resources/diagnosis/temp/mask"

    file = request.FILES.get("file")  # 获取文件对象，包括文件名文件大小和文件内容
    # 规定上传目录
    upload_path = "./resources/diagnosis/temp"
    # 判断文件夹是否存在
    folder = os.path.exists(upload_path)
    if not folder:
        os.makedirs(upload_path)
        print("创建文件夹")
    if file:
        file_name = file.name
        # 表示上传文件的后缀
        etx = ""
        # 判断文件是是否重名，重名了，文件名加时间
        if os.path.exists(os.path.join(upload_path, file_name)):
            name, etx = os.path.splitext(file_name)
            addtime = time.strftime("%Y%m%d%H%M%S")
            finally_name = name + "_" + addtime + etx
        else:
            finally_name = file.name

        # 文件分块上传
        upload_file_to = open(os.path.join(upload_path, finally_name), "wb+")
        for chunk in file.chunks():
            upload_file_to.write(chunk)
        upload_file_to.close()

        finally_name_no_etx, etx = os.path.splitext(finally_name)

        file_processed_url = os.path.join(
            upload_path, "res_" + finally_name_no_etx + ".png"
        )

        # 对图片进行分割并返回分割图片的base64
        img_base64 = seg_model(
            os.path.join(upload_path, finally_name), file_processed_url, mask_path
        )
    else:
        print("no file")
    print("finished")

    # 删除临时文件
    temp_name = "temp.png"
    os.remove(os.path.join(mask_path, temp_name))
    os.remove(os.path.join(upload_path, finally_name))
    os.remove(file_processed_url)
    return HttpResponse(img_base64, content_type="image/png")


def segExist(request):
    diagnosis_root = "./resources/diagnosis"

    img_id = request.GET.get("id")
    img_url = Patient.objects.filter(id=img_id)[0].url
    img_path = img_url.split("/", 3)[3]  #  resources/xxxxx.jpg
    img_name = img_path.split("/")[1]  # xxxx.jpg
    name, etx = os.path.splitext(img_name)
    seg_img_name = "res_" + name + ".png"
    seg_path = os.path.join(diagnosis_root, seg_img_name)
    if os.path.exists(seg_path):
        with open(seg_path, "rb") as f:
            image = f.read()
            image_base64 = base64.b64encode(image)
            return HttpResponse(image_base64, content_type="image/png")
    else:
        img_path = os.path.join("./", img_path)  # ./resources/xxxxx.jpg
        mask_path = "./resources/diagnosis/mask"
        img_base64 = seg_model(img_path, seg_path, mask_path)
        # 删除临时文件
        temp_name = "temp.png"
        os.remove(os.path.join(mask_path, temp_name))
        return HttpResponse(img_base64, content_type="image/png")


def gradImg(request):
    file = request.FILES.get("file")  # 获取文件对象，包括文件名文件大小和文件内容
    # 规定上传目录
    upload_path = "./resources/grading_diag"
    # 判断文件夹是否存在
    folder = os.path.exists(upload_path)
    if not folder:
        os.makedirs(upload_path)
        print("创建文件夹")
    if file:
        file_name = file.name
        # 表示上传文件的后缀
        etx = ""
        # 判断文件是是否重名，重名了，文件名加时间
        if os.path.exists(os.path.join(upload_path, file_name)):
            name, etx = os.path.splitext(file_name)
            addtime = time.strftime("%Y%m%d%H%M%S")
            finally_name = name + "_" + addtime + etx
        else:
            finally_name = file.name

        # 文件分块上传
        upload_file_to = open(os.path.join(upload_path, finally_name), "wb+")
        for chunk in file.chunks():
            upload_file_to.write(chunk)
        upload_file_to.close()

        # 对图片进行分割并返回分割图片的base64
        result = grading_model(os.path.join(upload_path, finally_name))
    else:
        print("no file")
    print("finished")

    # 删除临时文件
    os.remove(os.path.join(upload_path, finally_name))
    resp = {"isSuccess": True, "data": result}
    return HttpResponse(json.dumps(resp), content_type="application/json")


def gradExist(request):
    img_id = request.GET.get("id")
    img_url = Patient.objects.filter(id=img_id)[0].url
    img_path = img_url.split("/", 3)[3]  #  resources/xxxxx.jpg
    img_path = "./" + img_path

    result = grading_model(img_path)
    resp = {"isSuccess": True, "data": result}
    return HttpResponse(json.dumps(resp), content_type="application/json")
