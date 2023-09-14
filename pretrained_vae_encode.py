import os
import random
from torchvision.io import read_image
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pickle
import numpy as np
from einops import rearrange
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL
from itertools import count
import torch.nn.functional as F


def scan_dataset(img_dir):
    def count_subdirectories(path):
        subdirectories = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return len(subdirectories)

    current_directory = img_dir  # 当前文件夹路径，可以根据需要修改

    num_subdirectories = count_subdirectories(current_directory)
    print('current_directory:', current_directory)
    print('num_subdirectories:', num_subdirectories)

    fignum_1_case_num, fignum_2_case_num, fignum_3_case_num, fignum_4_case_num, fignum_5_case_num = 0, 0, 0, 0, 0
    for index in range(num_subdirectories):
        if index < 10:
            index_ = '00000' + str(index)
        elif index < 100:
            index_ = '0000' + str(index)
        elif index < 1000:
            index_ = '000' + str(index)
        elif index < 10000:
            index_ = '00' + str(index)
        elif index < 100000:
            index_ = '0' + str(index)
        elif index < 1000000:
            index_ = str(index)

        exis_path_0 = os.path.join(img_dir, index_, "rgb_top", "0.jpg")
        exis_path_1 = os.path.join(img_dir, index_, "rgb_top", "1.jpg")
        exis_path_2 = os.path.join(img_dir, index_, "rgb_top", "2.jpg")

        exis_path_3 = os.path.join(img_dir, index_, "rgb_top", "3.jpg")
        exis_path_4 = os.path.join(img_dir, index_, "rgb_top", "4.jpg")
        exis_path_5 = os.path.join(img_dir, index_, "rgb_top", "5.jpg")

        # FIXME   0 + 1 -> 2   0 + 2 -> 3
        #  todo 0 + 2 -> 1  ******

        # 2 + zero embedding ->0

        if os.path.exists(exis_path_5) == True:
            fignum_5_case_num += 1
        elif os.path.exists(exis_path_4) == True:
            fignum_4_case_num += 1
        elif os.path.exists(exis_path_3) == True:
            fignum_3_case_num += 1
        elif os.path.exists(exis_path_2) == True:
            fignum_2_case_num += 1
        elif os.path.exists(exis_path_1) == True:
            fignum_1_case_num += 1

    print('fignum_1_case_num:', fignum_1_case_num)
    print('fignum_2_case_num:', fignum_2_case_num)
    print('fignum_3_case_num:', fignum_3_case_num)
    print('fignum_4_case_num:', fignum_4_case_num)
    print('fignum_5_case_num:', fignum_5_case_num)
    return num_subdirectories



def load_vae_and_encode(img_dir):
    num_subdirectories = scan_dataset(img_dir)
    for index in range(num_subdirectories):
        if index < 10:
            index_ = '00000' + str(index)
        elif index < 100:
            index_ = '0000' + str(index)
        elif index < 1000:
            index_ = '000' + str(index)
        elif index < 10000:
            index_ = '00' + str(index)
        elif index < 100000:
            index_ = '0' + str(index)
        elif index < 1000000:
            index_ = str(index)


    exis_path_0 = os.path.join(img_dir, index_, "rgb_top", "0.jpg")
    exis_path_1 = os.path.join(img_dir, index_, "rgb_top", "1.jpg")
    exis_path_2 = os.path.join(img_dir, index_, "rgb_top", "2.jpg")
    exis_path_3 = os.path.join(img_dir, index_, "rgb_top", "3.jpg")
    exis_path_4 = os.path.join(img_dir, index_, "rgb_top", "4.jpg")
    exis_path_5 = os.path.join(img_dir, index_, "rgb_top", "5.jpg")
    img_labels_path = os.path.join(img_dir, index_, "trajectory.pkl")
    obs_path = os.path.join(img_dir, index_, "obs.pkl")

    print('exis_path_0:', exis_path_0) #/home/rl/COT-diffusion/pick_in_order_then_restore/002642/rgb_top/0.jpg

    # assert os.path.exists(exis_path_0) == False

    def count_subdirectories(path):
        files = os.listdir(path)  # 读入文件夹
        num_png = len(files)  # 统计文件夹中的文件个数
        return num_png

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to('cuda')

    current_directory = os.path.join(img_dir, index_, "rgb_top")  # 当前文件夹路径，可以根据需要修改
    # print('current_directory:', current_directory)
    num_png = count_subdirectories(current_directory)

    image_dict = {}
    unnorm_image_dict = {}
    for image_index in range(num_png):
        png_path = os.path.join(current_directory, "{}.jpg".format(image_index))

        image = read_image(png_path).to('cuda')
        # image = trans(image).unsqueeze(0).to('cuda')  # 对图片进行某些变换

        if self.transform is not None:
            image = self.transform(image)  # 对图片进行某些变换
        unnorm_image_dict[str(i)] = image
        image = image.unsqueeze(0).to('cuda')  # 对图片进行某些变换
        latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215)
        image_dict[str(i)] = latent_image
        # exit()



        # image = read_image(png_path).to('cuda')
        # print('image step 1:', image, image.shape)
        # # todo 这里可能要升维
        # image = trans(image).unsqueeze(0).to('cuda')  # 对图片进行某些变换
        # print('image step 2:', image, image.shape)
        # latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215)
        # latent_image_2 = vae.encode(image).latent_dist.sample()
        # print('latent_image:', latent_image, latent_image.shape)
        # reconstructed_image = vae.decode(latent_image / 0.18215).sample.squeeze(0)
        # reconstructed_image_2 = vae.decode(latent_image).sample.squeeze(0)
        # print('reconstructed_image:', reconstructed_image, reconstructed_image.shape)
        # loss = F.mse_loss(image, reconstructed_image, reduction='mean')
        # loss_2 = F.mse_loss(image, reconstructed_image_2, reduction='mean')
        # print('loss 1: ', loss)
        # print('loss 2: ', loss_2)
        # exit()








    image_dict = {}
    unnorm_image_dict = {}
    for i in range(num_subdirectories):
        read_image_path = os.path.join(self.img_dir, index_, "rgb_top", "{}.jpg".format(i))
        image = read_image(read_image_path)
        resize = transforms.Resize([32, 64])
        image = resize(image)
        unnorm_image_dict[str(i)] = image
        # print('unnorm_image in data :', image, image.shape)
        if self.transform is not None:
            image = self.transform(image)  # 对图片进行某些变换
        # print('image in data :', image, image.shape)

        # latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215)
        # print('latent_image:', latent_image, latent_image.shape)
        # exit()
        image_dict[str(i)] = image
        # exit()
        # image_list.append(image)





cur_dir = '/home/rl/COT-diffusion/pick_in_order_then_restore'

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


load_vae_and_encode(cur_dir)
# scan_dataset(cur_dir)




