from django.db import models

# Create your models here.

# 用户
class User(models.Model):
    account = models.CharField("账户", max_length=20)
    password = models.CharField("密码", max_length=15)


# 单条数据
class Patient(models.Model):
    # name = models.CharField('病人姓名', max_length=30, null=True, blank=True)
    gender = models.CharField("病人性别", max_length=5, null=True, blank=True)
    age = models.CharField("病人年龄", max_length=10, null=True, blank=True)
    position = models.CharField("影像部位", max_length=20)
    url = models.URLField("URL")


# 标注记录
class EyeAnnotation(models.Model):
    id = models.IntegerField("ID", primary_key=True)
    auto_annotation = models.CharField("自动标注结果", max_length=100, null=True, blank=True)
    manual_annotation = models.CharField(
        "手动标注结果", max_length=100, null=True, blank=True
    )
