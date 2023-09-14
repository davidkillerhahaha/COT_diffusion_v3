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
class myImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.img_labels = 2600  # 这是一个dataframe，0是文件名，1是类别
        self.imgs = 99999
        self.transform = transform

    def __len__(self):
        return self.img_labels  # 数据集长度

    def __getitem__(self, index):
        # 拼接得到图片文件路径
        # 例如img_dir为'D:/curriculum/2022learning/learnning_dataset/data/'
        # img_labels.iloc[index, 0]为5.jpg
        # 那么img_path为'D:/curriculum/2022learning/learnning_dataset/data/5.jpg'


########################################################################################################################
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

                exis_path_0 = os.path.join(self.img_dir, index_, "rgb_top", "0.jpg")
                exis_path_1 = os.path.join(self.img_dir, index_, "rgb_top", "1.jpg")
                exis_path_2 = os.path.join(self.img_dir, index_, "rgb_top", "2.jpg")

                exis_path_3 = os.path.join(self.img_dir, index_, "rgb_top", "3.jpg")
                exis_path_4 = os.path.join(self.img_dir, index_, "rgb_top", "4.jpg")
                exis_path_5 = os.path.join(self.img_dir, index_, "rgb_top", "5.jpg")

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

        # scan_dataset(self.img_dir)
        # todo  fignum_2_case_num: 45814 fignum_3_case_num: 5241 fignum_4_case_num: 16 （for rearrange）

        # exit()

########################################################################################################################

        while True:
            index += 1
            index = index % 2600
            # index = random.randint(0, 2600)
            if index<10:
                index_ = '00000' + str(index)
            elif index< 100:
                index_ = '0000' + str(index)
            elif index < 1000:
                index_ = '000' + str(index)
            elif index < 10000:
                index_ = '00' + str(index)
            elif index < 100000:
                index_ = '0' + str(index)
            elif index < 1000000:
                index_ = str(index)

            # print('index_', index_)

            exis_path_0 = os.path.join(self.img_dir, index_, "rgb_top", "0.jpg")
            exis_path_1 = os.path.join(self.img_dir, index_, "rgb_top", "1.jpg")
            exis_path_2 = os.path.join(self.img_dir, index_, "rgb_top", "2.jpg")
            exis_path_3 = os.path.join(self.img_dir, index_, "rgb_top", "3.jpg")
            exis_path_4 = os.path.join(self.img_dir, index_, "rgb_top", "4.jpg")
            exis_path_5 = os.path.join(self.img_dir, index_, "rgb_top", "5.jpg")
            img_labels_path = os.path.join(self.img_dir, index_, "trajectory.pkl")
            obs_path = os.path.join(self.img_dir, index_, "obs.pkl")

            if os.path.exists(exis_path_0) == False or os.path.exists(exis_path_4) == True:
                continue

            # with open(img_labels_path, "rb") as f:
            #     traj = pickle.load(f)
            #     print('traj: ', traj)

            def count_subdirectories(path):
                files = os.listdir(path)  # 读入文件夹
                num_png = len(files)  # 统计文件夹中的文件个数
                return num_png

            current_directory = os.path.join(self.img_dir, index_, "rgb_top")  # 当前文件夹路径，可以根据需要修改
            # print('current_directory:', current_directory)
            num_subdirectories = count_subdirectories(current_directory)
            # if num_subdirectories == 5:
            #     # print('num_subdirectories:', num_subdirectories)
            #     print('This case has {} images in one folder.'.format(num_subdirectories))
            #     # with open(obs_path, "rb") as f:
            #     #     obs = pickle.load(f)
            #     #     print('obs:', obs, obs['segm']['top'].shape)
            #     #     objects, ee = obs["objects"], obs["ee"]
            #     #     print('objects:', obs['objects'], obs['objects'].shape)
            #     image = read_image(exis_path_2)
            #     # print('image:', image, image.shape)
            #     # exit()
            #     # envs.obs  --> (3, 128, 256)
            #     # obs.pkl  --> (1, 128, 256)  ee(5)

            # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to('cuda')
            # vae = AutoencoderKL.from_pretrained('/home/nifei/sd-vae-ft-ema')
            vae = AutoencoderKL.from_pretrained('/home/nifei/stable_diffusion_pt')

            if vae is not None:
                vae.requires_grad_(False)

            # vae = AutoencoderKL.from_pretrained(
            #     'CompVis/stable-diffusion-v1-4', subfolder="vae"
            # )

            # vae = AutoencoderKL.from_pretrained('/home/nifei/sd-vae-ft-ema').to(self.device)

            # image = read_image(png_path).to('cuda')
            # image = trans(image).unsqueeze(0).to('cuda')  # 对图片进行某些变换
            # latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215)
            # if self.transform is not None:
            #     image = self.transform(image)  # 对图片进行某些变换
            #
            image_dict = {}
            unnorm_image_dict = {}
            for i in range(num_subdirectories):
                read_image_path = os.path.join(self.img_dir, index_, "rgb_top", "{}.jpg".format(i))
                image = read_image(read_image_path)
                unnorm_image_dict[str(i)] = image
                if self.transform is not None:
                    image = self.transform(image)  # 对图片进行某些变换
                # resize = transforms.Resize([32, 64])
                # image = trans(image).unsqueeze(0).to('cuda')  # 对图片进行某些变换
                # image = image.unsqueeze(0).to(self.device)  # 对图片进行某些变换
                image = image.unsqueeze(0) # 对图片进行某些变换
                # print('unnorm_image in data :', image, image.shape)
                # print('image in data :', image, image.shape)
                latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215).squeeze(0)
                # print('latent_image:', latent_image, latent_image.shape)
                # exit()
                image_dict[str(i)] = latent_image
                # exit()
                # image_list.append(image)

            break

            # image_dict = {}
            # unnorm_image_dict = {}
            # for i in range(num_subdirectories):
            #     read_image_path = os.path.join(self.img_dir, index_, "rgb_top", "{}.jpg".format(i))
            #     image = read_image(read_image_path)
            #     resize = transforms.Resize([32, 64])
            #     image = resize(image)
            #     unnorm_image_dict[str(i)] = image
            #     # print('unnorm_image in data :', image, image.shape)
            #     if self.transform is not None:
            #         image = self.transform(image)  # 对图片进行某些变换
            #     # print('image in data :', image, image.shape)
            #
            #     latent_image = vae.encode(image).latent_dist.sample().mul_(0.18215)
            #     # print('latent_image:', latent_image, latent_image.shape)
            #     # exit()
            #     image_dict[str(i)] = image
            #     # exit()
            #     # image_list.append(image)
            #
            # break








        return unnorm_image_dict, image_dict,  img_labels_path,  num_subdirectories
        # return image_dict, img_labels_path, obs_path,  num_subdirectories
        # return image, img_labels_path
        # return image_list, img_labels_path



            # if os.path.exists(exis_path_5) == True:
            #     print('This case has 5 images in one folder.')
            # elif os.path.exists(exis_path_4) == True:
            #     print('This case has 4 images in one folder.')
            # elif os.path.exists(exis_path_3) == True:
            #     print('This case has 3 images in one folder.')
            # elif os.path.exists(exis_path_2) == True:
            #     print('This case has 2 images in one folder.')
            # elif os.path.exists(exis_path_1) == True:
            #     print('This case has 1 images in one folder.')


            # with open(obs_path, "rb") as f:
            #     obs = pickle.load(f)
            #     print('obs: ', obs)
            # with open(img_labels_path, "rb") as f:
            #     traj = pickle.load(f)
            #     print('traj: ', traj)

            # exit()

            # # todo 这里视任务而定
            # if os.path.exists(exis_path_0) == True and os.path.exists(exis_path_2) == True:
            #     inital_image = read_image(exis_path_0)
            #     # todo 这里有问题，挑选的应该是2帧内
            #     image_0 = read_image(exis_path_1)  # tensor类型
            #     image_1 = read_image(exis_path_2)
            #     resize = transforms.Resize([32, 64])
            #     image_0 = resize(image_0)
            #     image_1 = resize(image_1)
            #     if self.transform is not None:
            #         image_0 = self.transform(image_0)  # 对图片进行某些变换
            #         image_1 = self.transform(image_1)
            #     break

        # return image_0, image_1, img_labels_path

        # for i in count():
        #     if os.path.exists(exis_path) == True and os.path.exists(exis_path_1) == False :
        #         break
        #     index +=1
        #     if index >= self.imgs:
        #         index = 0
        #     if index < 10:
        #         index_ = '00000' + str(index)
        #     elif index < 100:
        #         index_ = '0000' + str(index)
        #     elif index < 1000:
        #         index_ = '000' + str(index)
        #     elif index < 10000:
        #         index_ = '00' + str(index)
        #     elif index < 100000:
        #         index_ = '0' + str(index)
        #     elif index < 1000000:
        #         index_ = str(index)
        #     exis_path = os.path.join(self.img_dir, index_, "rgb_top", "2.jpg")
        #     exis_path_1 = os.path.join(self.img_dir, index_, "rgb_top", "3.jpg")
        # img_labels_path = os.path.join(self.img_dir, index_, "trajectory.pkl")
        # '''
        # with open(img_labels_path, 'rb') as pickle_file:
        #     img_label = pickle.load(pickle_file)
        # label = img_label['prompt']
        # label_2 = img_label['prompt_assets']
        # '''
        # conditional_img_path = os.path.join(self.img_dir, index_,"rgb_top", "0.jpg")
        # img_path_0 = os.path.join(self.img_dir, index_, "rgb_top", "0.jpg")
        # img_path_1 = os.path.join(self.img_dir, index_, "rgb_top", "1.jpg")
        # image_0 = read_image(img_path_0)  # tensor类型
        # image_1 = read_image(img_path_1)
        # resize = transforms.Resize([32, 64])
        # image_0 = resize(image_0)
        # image_1 = resize(image_1)
        # #print("zzzzzz",image.size())
        #
        #
        # if self.transform is not None:
        #     image_0 = self.transform(image_0)  # 对图片进行某些变换
        #     image_1 = self.transform(image_1)
        # path = img_labels_path
        # '''
        # with open(os.path.join(path, "obs.pkl"), "rb") as f:
        #     obs = pickle.load(f)
        #
        # rgb_dict = {"front": [], "top": []}
        # n_rgb_frames = len(os.listdir(os.path.join(path, f"rgb_front")))
        # for view in ["front", "top"]:
        #     for idx in range(n_rgb_frames):
        #         # load {idx}.jpg using PIL
        #         rgb_dict[view].append(
        #             rearrange(
        #                 np.array(
        #                     Image.open(os.path.join(path, f"rgb_{view}", f"{idx}.jpg")),
        #                     copy=True,
        #                     dtype=np.uint8,
        #                 ),
        #                 "h w c -> c h w",
        #             )
        #         )
        # rgb_dict = {k: np.stack(v, axis=0) for k, v in rgb_dict.items()}
        # segm = obs.pop("segm")
        # end_effector = obs.pop("ee")
        #
        # with open(os.path.join(path, "action.pkl"), "rb") as f:
        #     action = pickle.load(f)
        #
        #
        # with open(path, "rb") as f:
        #     traj_meta = pickle.load(f)
        #
        # prompt = traj_meta.pop("prompt")
        # prompt_assets = traj_meta.pop("prompt_assets")
        # #print("sss", prompt_assets)
        #
        # for k, v in rgb_dict.items():
        #     print(f"RGB {k} view : {v.shape}")
        # for k, v in segm.items():
        #     print(f"Segm {k} view : {v.shape}")
        # print("End effector : ", end_effector.shape)
        # print("-" * 50)
        # print("Action")
        # for k, v in action.items():
        #     print(f"{k} : {v.shape}")
        # print("-" * 50)
        # print("Prompt: ", prompt)
        # print("Prompt assets keys: ", str(list(prompt_assets.keys())))
        # '''
        # #print("SSS", image_0.size(),image_1.size())
        #
        #
        #
        #
        #
        #
        # return image_0, image_1, path#prompt, prompt_assets

