# -*- coding: utf-8 -*-
import os, os.path
import cv2
import torch
from PIL import Image
import numpy as np

img = Image.open("./data/IM1.jpg")

print(img.size)

img = img.resize((100,100), Image.ANTIALIAS)
print(img.size)

image_sort = ["png", "jpg", "jpeg"]
image_list = []

path = "./data"

for filename in os.listdir(path):
    for i in image_sort:
        if filename.lower().endswith(i):
            file_path = path + "/" + filename
            image = cv2.imread(file_path)
            image_list.append(image)

print(image_list)