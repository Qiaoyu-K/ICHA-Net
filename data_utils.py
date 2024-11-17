from pathlib import Path

import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import sys
import cv2

sys.path.append('.')
sys.path.append('..')
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt


def tensorShow(tensors, titles=None):
    '''
    t:BCWH
    '''
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=opt.resize_size, format='.png'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index].replace('\\', '/')
        id = img.split('/')[-1].split('_')[0]
        clear_name = id + self.format
        if opt.datasets == 'NH' or opt.datasets == 'DENSE':
            clear_name = id + '_' + 'GT' + self.format
        elif opt.datasets == '6K':
            self.format = img.split('/')[-1].split('.')[-1]
            if img.split('/')[-3] == 'eval':
                clear_name = id + '.png'
            else:
                clear_name = id + '.' + self.format
        elif opt.datasets == 'OTS':
            img = self.haze_imgs[index].replace('\\', '/')
            id = img.split('/')[-1].split('_')[0]
            clear_name = id + '.jpg'
        elif opt.datasets == 'OHAZE':
            hazy_name = Path(img).name
            clear_name = hazy_name.replace('hazy', 'GT')
        else:
            print('输入数据名称错误')
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)

        if not isinstance(self.size, str):
            haze = cv2.resize(np.array(haze), dsize=(self.size, self.size))
            clear = cv2.resize(np.array(clear), dsize=(self.size, self.size))
            haze = Image.fromarray(haze)
            clear = Image.fromarray(clear)

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))

        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)


import os

BS = opt.bs
print('Batchsize:', BS)
path = opt.path

ITS_train_loader = DataLoader(dataset=RESIDE_Dataset(path + 'train', train=True, size=opt.resize_size), batch_size=BS,
                              shuffle=True)
ITS_test_loader = DataLoader(dataset=RESIDE_Dataset(path + 'eval', train=False, size=opt.resize_size),
                             batch_size=1, shuffle=False)
print('train_data:', len(ITS_train_loader), 'eval_data:', len(ITS_test_loader))

if __name__ == "__main__":
    pass
