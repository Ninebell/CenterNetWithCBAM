from tqdm import tqdm
import cv2
import argparse
import torch
import shutil
import os
import random

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import torch.nn as nn

import numpy as np

from torch_model.losses import custom_center_loss

import torch_model.center_net
from __generator import data_generator


def rect_prism(x_range, y_range, z_range, ax, color):
    Z = [
        [x_range[0], y_range[0], z_range[0]],
        [x_range[0], y_range[0], z_range[1]],
        [x_range[0], y_range[1], z_range[0]],
        [x_range[0], y_range[1], z_range[1]],
        [x_range[1], y_range[0], z_range[0]],
        [x_range[1], y_range[0], z_range[1]],
        [x_range[1], y_range[1], z_range[0]],
        [x_range[1], y_range[1], z_range[1]],
    ]
    verts = [[Z[0], Z[2], Z[4], Z[6]],
             [Z[1], Z[3], Z[5], Z[7]],

             [Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],

             [Z[0], Z[1], Z[4], Z[5]],
             [Z[2], Z[3], Z[6], Z[7]],
             ]

    # plot sides

    ax.add_collection3d(Poly3DCollection(verts,
                                         facecolors=color, linewidths=1, edgecolors=color, alpha=.25))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # x_y_edge(x_range, y_range, z_range)
    # y_z_edge(x_range, y_range, z_range)
    # x_z_edge(x_range, y_range, z_range)


def draw_roi(img, heat, size):

    heat = np.squeeze(heat)
    size = np.squeeze(size)
    heat = np.reshape(heat,(64,64))
    img = np.asarray(img*255,dtype=np.uint8)
    img = Image.fromarray(img)
    img_draw = ImageDraw.Draw(img)

    center = []
    for r in range(1,63):
        for c in range(1,63):
            if heat[r,c] == np.max(heat[r-1:r+2, c-1:c+2]) and heat[r,c] > 0.5:
                center.append((c,r))
    points = []
    for point in center:
        w = size[0,point[1],point[0]] / 592 * 256
        h = size[1,point[1],point[0]] / 480 * 256
        z = size[2,point[1],point[0]]
        point = point[0]*4, point[1]*4, z
        points.append([w,h,point])
        img_draw.rectangle((point[0]-w//2, point[1]-h//2, point[0]+w//2, point[1]+h//2), outline='red', width=2)

    return img, points


def train_model(net, optim, criterion, batch_size, is_cuda=True):
    iter_count = 0
    epoch_loss = 0
    repeat = net.n_stack
    for data in tqdm(data_generator(batch_size, shuffle=True, is_train=True)):
        iter_count += 1
        x, seg = data
        optim.zero_grad()
        if is_cuda:
            x = torch.from_numpy(x).type(torch.FloatTensor).cuda(non_blocking=True)
            seg = torch.from_numpy(seg).type(torch.FloatTensor).cuda(non_blocking=True)

        result = net(x)

        loss = 0
        for i in range(repeat):
            inter_loss = criterion(result[i][0], seg)
            loss += inter_loss
        loss.backward()
        epoch_loss += loss.item()
        optim.step()
    return epoch_loss, iter_count


def test_model(net, is_cuda, save_path):
    i = 0
    target_img = np.zeros((256,256,3))
    predict_img = np.zeros((256,256,3))
    input_img = np.zeros((256,256,3))
    cv2.namedWindow("input")
    cv2.namedWindow("target")
    cv2.namedWindow("predict")
    vc = cv2.VideoCapture("E:\\dataset\\Droplet\\video\\C001H001S0003 (1).avi")
    # for data in data_generator(1, False, False):
    #     x, seg = data
    while True:
        ret, x = vc.read()
        cv2.imshow("T", x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (256, 256))
        x = np.moveaxis(x, -1, 0)
        x = np.expand_dims(x, 0)/255.
        print(x.shape)
        if is_cuda:
            x = torch.from_numpy(x).type(torch.FloatTensor).cuda()

        with torch.no_grad():
            result = net(x)

            input_img = x[0].cpu().numpy()
            # seg_img = seg[0]

            input_img = np.moveaxis(input_img, 0, -1)
            input_img = np.array(input_img*255, dtype=np.uint8)
            input_img = cv2.cvtColor(input_img,cv2.COLOR_RGB2BGR)

            show = cv2.resize(input_img, (512, 512))

            cv2.imshow("input", show)

            # seg_img = np.array(np.reshape(np.where(seg_img>0.5, 1, 0), (64, 64, 1))*255,dtype=np.uint8)
            # seg_img = np.tile(seg_img, 3)
            # # seg_img.reshape((256,256,3))
            # seg_img = cv2.resize(seg_img, (256,256))
            # target_img = np.array(input_img & seg_img, dtype=np.uint8)
            # show = cv2.resize(target_img, (512, 512))
            # cv2.imshow("target", show)
            cv2.waitKey(10)


            # plt.figure(figsize=(6.4, 7.2))
            #
            # plt.subplot(2, 2, 1)
            # plt.imshow(input_img)
            #
            # plt.subplot(2, 2, 3)
            # plt.imshow(seg_img)
            #
            with torch.no_grad():
                seg_img_pred = result[-1][0].cpu().numpy()
                seg_img_pred = np.reshape(seg_img_pred, (64, 64))
                seg_img = np.array(np.reshape(np.where(seg_img_pred > 0.5, 1, 0), (64, 64, 1)) * 255, dtype=np.uint8)
                seg_img = np.tile(seg_img, 3)
                seg_img = cv2.resize(seg_img, (256, 256))
                target_img = np.array(input_img & seg_img, dtype=np.uint8)
                show = cv2.resize(target_img, (512, 512))
                cv2.imshow("predict", show)
                cv2.waitKey(0)

                # plt.subplot(2, 2, 4)
                # plt.imshow(seg_img_pred)

            # plt.savefig(save_path+"/{0}.png".format(i))
            plt.show()
            plt.close()

            i = i + 1


def print_train_info(epoch, batch_size):
    print()
    print("{0:^40s}".format('Train Information'))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('epoch', epoch)))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('batch size', batch_size)))
    # print("{0:^40s}".format("{0:22s}: {1:10,d}".format('input data', conf.get_train_data_num())))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', '-r', nargs='+', help='hourglass count', default=[2], dest='repeat', type=int)
    parser.add_argument('--nstack', '-n', nargs='+', help='hourglass layer count', default=[4], dest='n_stack', type=int)
    parser.add_argument('--epoch', '-e', nargs='+', help='epoch count', default=[5000], dest='epoch', type=int)
    parser.add_argument('--batch', '-b', nargs='+', help='batch size', default=[8], dest='batch_size', type=int)
    parser.add_argument('--pretrain', '-p', nargs='+', help='pretrain model', default=[None], dest='pretrain')
    parser.add_argument('--root_path', '-rp', nargs='+', help='save root path', default=['E:\\dataset\\Droplet'], dest='root_path')

    repeat = parser.parse_args().repeat
    n_stack = parser.parse_args().n_stack
    epoch = parser.parse_args().epoch
    batch_size = parser.parse_args().batch_size
    pretrain = parser.parse_args().pretrain
    root_path = parser.parse_args().root_path
    return epoch[0], batch_size[0], repeat[0], n_stack[0], pretrain[0], root_path[0]


def _main(epoches, batch_size, repeat, n_layer, pretrain, root_path):
    min_loss = 10000

    net = torch_model.center_net.CenterNet(256, [1], out_activation=[torch.sigmoid],n_layer=n_layer, n_stack=repeat)
    net.info()
    print_train_info(epoches, batch_size)

    criterion = nn.BCEWithLogitsLoss()

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        net = net.cuda()

    lr = 1e-3

    if pretrain is not None:
        net.load_state_dict(torch.load(pretrain))

    optim = torch.optim.Adam(net.parameters(), lr)

    sch = torch.optim.lr_scheduler.StepLR(optim, 50)

    # for epoch in range(1, epoches):
    #     epoch_loss, iter_count = train_model(net, optim, criterion, batch_size)
    #     epoch_loss /= iter_count
    #     # sch.step()
    #
    #     print()
    #     print('\n', epoch, epoch_loss, '\n')
    #     print()
    #     if epoch > 100 and epoch_loss < min_loss:
    #         save_path = '{0}\\result\\{1}_{2:4d}\\'.format(root_path, epoch, int(epoch_loss * 100))
    #         os.makedirs(save_path, exist_ok=True)
    #         torch.save(net.state_dict(), '{0}\\model.dict'.format(save_path))
    #         min_loss = epoch_loss
    #         test_model(net, is_cuda,save_path)

    save_path = '{0}\\result\\last'.format(root_path)
    net.load_state_dict(torch.load('{0}\\result\\model.dict'.format(root_path)))
    test_model(net, is_cuda,save_path)


if __name__ == "__main__":
    epoch, batch_size, repeat, n_stack, pretrain, root_path = get_arguments()
    _main(epoch, batch_size, repeat, n_stack, pretrain, root_path)

if __name__ == "__main__2":
    base_path = 'E:\\dataset\\Droplet\\'
    file_list = os.listdir(base_path+'label')
    random.shuffle(file_list)
    length = len(file_list)
    train_length = int(length*0.7)
    for j in range(length):
        if j < train_length:
            shutil.copy2(os.path.join(base_path+'label\\', file_list[j]),os.path.join(base_path+'train\\label', file_list[j]))
            shutil.copy2(os.path.join(base_path+'input\\', file_list[j]),os.path.join(base_path+'train\\input', file_list[j]))
        else:
            shutil.copy2(os.path.join(base_path+'label\\', file_list[j]),os.path.join(base_path+'validate\\label', file_list[j]))
            shutil.copy2(os.path.join(base_path+'input\\', file_list[j]),os.path.join(base_path+'validate\\input', file_list[j]))
