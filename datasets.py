import os

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import transforms

import utils
from PIL import Image
from preprocessing.construct_3dvg import graph_construct, build_graph_with_npy_features


class MyCAERDataset(Dataset):
    def __init__(self, root, train='train', transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.sample_folders = self.get_sample_folder()
        self.label = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def get_sample_folder(self):
        data_dir = ''
        if self.train == 'train':
            data_dir = os.path.join(self.root, 'train')
        elif self.train == 'test':
            data_dir = os.path.join(self.root, 'test')
        elif self.train == 'test':
            data_dir = os.path.join(self.root, 'val')
        else:
            raise TypeError("Mode {} not exits, please select mode in : ['train, 'test', 'val'].")
        labels = os.listdir(data_dir)
        data_dir_ls = []
        for label in labels:
            img_folders = os.listdir(os.path.join(data_dir, label))
            for img_folder in img_folders:
                data_dir_ls.append(os.path.join(os.path.join(data_dir, label), img_folder))
        return data_dir_ls

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, index):
        if index >= len(self.sample_folders):
            raise IndexError("Index out of bound.")
        img_folder = self.sample_folders[index]
        data = {}
        label = self.label.index(img_folder.split('/')[-2])
        img_files, npy_files, txt_files = utils.categorize_files(img_folder)
        obj_info_ls = utils.read_object_info_from_txt(os.path.join(img_folder, txt_files))
        agent_index = -1
        for img_file in img_files:
            if 'face' in img_file:
                data['face'] = Image.open(os.path.join(img_folder, img_file))
                agent_index = int(img_file.split("_")[1].split[0])
        mask_path = None
        for img_file in img_files:
            if 'mask' in img_file:
                mask_path = os.path.join(img_folder, img_file)
                origin_img_path = os.path.join(img_folder, f'person_{agent_index}.png')
                mask_img = utils.apply_mask_to_image(mask_path, origin_img_path)
                data['pose'] = mask_img
        for img_file in img_files:
            if 'origin' in img_file:
                agent_info = obj_info_ls[agent_index]
                area = [agent_info.x1, agent_info.x2, agent_info.width, agent_info.height]
                data['bg_context'] = utils.apply_mask_to_specific_area(os.path.join(img_folder, img_file), mask_path, area)
        depth_path = ''
        for img_file in img_files:
            if 'depth' in img_file:
                depth_path = os.path.join(img_folder, img_file)

        agent_info = obj_info_ls[agent_index]
        depth_img = Image.open(depth_path)
        depth_img = np.array(depth_img)
        # read csv data
        csv_path = os.path.join(img_folder, 'processed.csv')
        gaze_angle = utils.read_gaze_angles(csv_path)
        weights = graph_construct(obj_info_ls, agent_index, gaze_angle, depth_img)
        # get features
        graph = build_graph_with_npy_features(agent_index, npy_files, weights)
        data['3dvg'] = graph
        if self.transform:
            data['face'] = self.transform(data['face'])
            data['pose'] = self.transform(data['pose'])
            data['bg_context'] = self.transform(data['bg_context'])
        return data, torch.tensor(label, dtype=torch.float32)



class Emotic_CSVDataset(Dataset):

    def __init__(self, data_df, cat2ind, transform, data_src='./', train='train'):
        super(Emotic_CSVDataset, self).__init__()
        self.data_df = data_df
        if train == 'train':
            self.data_src = os.path.join(data_src, 'train')
        elif train == 'test':
            self.data_src = os.path.join(data_src, 'test')
        else:
            self.data_src = os.path.join(data_src, 'val')
        self.data_src = data_src
        self.transform = transform
        self.cat2ind = cat2ind
        # self.context_norm = transforms.Normalize(context_norm[0], context_norm[
        #     1])  # Normalizing the context image with context mean and context std
        # self.body_norm = transforms.Normalize(body_norm[0],
        #                                       body_norm[1])  # Normalizing the body image with body mean and body std

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.loc[index]
        # image_context = Image.open(os.path.join(self.data_src, row['Folder'], row['Filename']))
        data_folder = os.path.join(self.data_src, row['Folder'], row['Filename'])
        data = {}
        label = data_folder.split('/')[-2]
        img_files, npy_files, txt_files = utils.categorize_files(data_folder)
        obj_info_ls = utils.read_object_info_from_txt(os.path.join(data_folder, txt_files))
        agent_index = -1
        for img_file in img_files:
            if 'face' in img_file:
                data['face'] = Image.open(os.path.join(data_folder, img_file))
                agent_index = int(img_file.split("_")[1].split[0])
        mask_path = None
        for img_file in img_files:
            if 'mask' in img_file:
                mask_path = os.path.join(data_folder, img_file)
                origin_img_path = os.path.join(data_folder, f'person_{agent_index}.png')
                mask_img = utils.apply_mask_to_image(mask_path, origin_img_path)
                data['pose'] = mask_img
        for img_file in img_files:
            if 'origin' in img_file:
                agent_info = obj_info_ls[agent_index]
                area = [agent_info.x1, agent_info.x2, agent_info.width, agent_info.height]
                data['bg_context'] = utils.apply_mask_to_specific_area(os.path.join(data_folder, img_file), mask_path,
                                                                       area)
        depth_path = ''
        for img_file in img_files:
            if 'depth' in img_file:
                depth_path = os.path.join(data_folder, img_file)

        agent_info = obj_info_ls[agent_index]
        depth_img = Image.open(depth_path)
        depth_img = np.array(depth_img)
        # read csv data
        csv_path = os.path.join(data_folder, 'processed.csv')
        gaze_angle = utils.read_gaze_angles(csv_path)
        weights = graph_construct(obj_info_ls, agent_index, gaze_angle, depth_img)
        # get features
        graph = build_graph_with_npy_features(agent_index, npy_files, weights)
        data['3dvg'] = graph
        if self.transform:
            data['face'] = self.transform(data['face'])
            data['pose'] = self.transform(data['pose'])
            data['bg_context'] = self.transform(data['bg_context'])
        cat_labels = ast.literal_eval(row['Categorical_Labels'])
        cont_labels = ast.literal_eval(row['Continuous_Labels'])
        one_hot_cat_labels = self.cat_to_one_hot(cat_labels)
        return data, torch.tensor(one_hot_cat_labels, dtype=torch.float32), torch.tensor(
            cont_labels, dtype=torch.float32) / 10.0

    def cat_to_one_hot(self, cat):
        one_hot_cat = np.zeros(26)
        for em in cat:
            one_hot_cat[self.cat2ind[em]] = 1
        return one_hot_cat