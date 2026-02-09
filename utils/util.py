from PIL import Image
import torch
import os
import numpy as np
import sys
import argparse
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms


def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)


# get pre-resized 84x84 images for validation and test
def get_pre_folder(image_folder, transform_type):
    split = ['val', 'test']

    if transform_type == 0:
        transform = transforms.Compose([transforms.Resize(92),
                                        transforms.CenterCrop(84)])
    elif transform_type == 1:
        transform = transforms.Compose([transforms.Resize([92, 92]),
                                        transforms.CenterCrop(84)])

    cat_list = []

    for i in split:

        cls_list = os.listdir(os.path.join(image_folder, i))

        folder_name = i + '_pre'

        mkdir(os.path.join(image_folder, folder_name))

        for j in tqdm(cls_list):

            mkdir(os.path.join(image_folder, folder_name, j))

            img_list = os.listdir(os.path.join(image_folder, i, j))

            for img_name in img_list:
                img = Image.open(os.path.join(image_folder, i, j, img_name))
                img = img.convert('RGB')
                img = transform(img)
                img.save(os.path.join(image_folder, folder_name, j, img_name[:-3] + 'png'))


def get_device_map(gpu):
    cuda = lambda x: 'cuda:%d' % x
    temp = {}
    for i in range(5):
        temp[cuda(i)] = cuda(gpu)
    return temp


# def get_device_map(gpus):
#     """
#     根据传入的 GPU 列表生成设备映射
#     :param gpus: list，表示需要映射的 GPU 设备，例如 [0, 1, 2] -> [2, 3, 4]
#     :return: 设备映射的字典
#     """
#     cuda = lambda x: 'cuda:%d' % x  # lambda 函数用于生成 'cuda:X' 形式的字符串
#     temp = {}  # 初始化一个空字典
#
#     # 确保 gpus 列表非空，并且不超出系统 GPU 数量
#     assert len(gpus) > 0, "必须至少传入一个 GPU 设备编号"
#
#     for i, gpu in enumerate(gpus):
#         temp[cuda(i)] = cuda(gpu)  # 将每个设备按顺序映射到传入的目标 GPU
#     return temp
#

