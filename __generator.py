import os
import json
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageOps

# base_dir = 'C:\\Users\\NR\\Desktop\\test'
base_dir = 'D:\\Users\\NR\\Desktop\\test'
base_dir = 'E:\\dataset\\Droplet'
# seg_dir = '{0}\\seg'.format(base_dir)
# bottom_tag = 'b.png'


def make_heat(shape, center):
    base = np.zeros(shape)
    base[center[1],center[0]] = 1
    sigma = 1
    gaus = gaussian_filter(base, sigma)
    gaus = gaus/np.max(gaus)
    return gaus


def data_generator(batch_size, shuffle, is_train=False):
    def read_img(path, shape, mode='RGB'):
        img = Image.open(path).resize(shape)
        if mode == 'L':
            img = img.convert(mode)
        img = np.asarray(img)
        return img/255.
    if is_train:
        root_path = base_dir + '\\train'
    else:
        root_path = base_dir + '\\validate'

    input_dir = '{0}\\input'.format(root_path)
    label_dir = '{0}\\label'.format(root_path)
    labels = os.listdir(label_dir)

    if shuffle:
        random.shuffle(labels)

    iters = len(labels) // batch_size

    for idx in range(iters):
        x = []
        seg_list = []

        for b in range(batch_size):
            file_name = labels[idx*batch_size+b]
            label_path = '{0}\\{1}'.format(label_dir, file_name)
            input_path = '{0}\\{1}'.format(input_dir, file_name)
            input_img = read_img(input_path, (256, 256))
            label_img = read_img(label_path, (64, 64), 'L')

            input_img = np.moveaxis(input_img, 2, 0)
            label_img = np.expand_dims(label_img, -1)
            label_img = np.moveaxis(label_img, 2, 0)
            # label_img = np.where(label_img>0.5, True, False)
            x.append(input_img)
            seg_list.append(label_img)
        x = np.asarray(x)
        seg_list = np.asarray(seg_list)
        yield x, seg_list
