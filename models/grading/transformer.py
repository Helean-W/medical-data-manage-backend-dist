import cv2
import numbers
from PIL import ImageOps

class CenterCrop_np(object):
    def __init__(self, size, padding=0):
        self.padding = padding
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.shape[:2]
        th, tw = self.size

        if w == tw and h == th:
            return img
        if w < tw or h < th:
            return cv2.resize(img, (tw, th))

        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return img[x1: x1 + tw, y1: y1 + th, :]


class Scale(object):
    def __init__(self, size_w, size_h):
        self.size_w = size_w
        self.size_h = size_h

    def __call__(self, img):
        w, h = img.shape[:2]
        if (w >= h and w == self.size_w) or (h >= w and h == self.size_h):
            return img
        if w > h:
            ow = self.size_w
            oh = int(self.size_h * h / w)
            return cv2.resize(img, (ow, oh))
        else:
            ow = int(self.size_w * w / h)
            oh = self.size_h
            return cv2.resize(img, (ow, oh))