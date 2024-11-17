import cv2
import numpy as np
from PIL import Image
from models import *
import torch
import torchvision.transforms as tfs
import metrics
from option import opt

abs = os.getcwd() + '/'
abs_data = opt.path

gps = opt.gps
blocks = opt.blocks
dataset = opt.task

img_dir = abs_data + opt.test_imgs + '/'
img_clear_dir = abs_data + opt.clear_imgs
output_dir = abs + f'pred_Network_{dataset}' + '_' + opt.datasets + '/'
print("pred_dir:", output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir = opt.model_dir
device = opt.device
ckp = torch.load(model_dir, map_location=device)

net = Network(dim=3, kernel_size=3, blocks=opt.blocks)
net.load_state_dict(ckp['model'])
net.to(opt.device).eval()

ssim_all = []
psnr_all = []
for index, im in enumerate(os.listdir(img_dir)):
    print(f'\r {im}', end='', flush=True)
    if opt.datasets == 'OHAZE':
        haze = Image.open(img_dir + im)
        clear_name = im.replace('hazy', 'GT')
        clear = Image.open(img_clear_dir + clear_name)

    else:
        im_num = im.split('_')[0]
        haze = Image.open(img_dir + im)
        im_format = im.split('_')[-1].split('.')[-1]
        if opt.datasets == 'NH' or opt.datasets == 'DENSE':
            clear = Image.open(img_clear_dir + im_num + '_' + 'GT' + '.png')
        else:
            clear = Image.open(img_clear_dir + im_num + '.png')
        if opt.datasets == 'ITS' or im_format == 'png':
            clear_no = np.array(clear)
            a = clear_no.shape[0]
            b = clear_no.shape[1]
            clear = clear_no[10:a - 10, 10:b - 10, ...]

    haze = cv2.resize(np.array(haze), dsize=(opt.resize_size, opt.resize_size))
    clear = cv2.resize(np.array(clear), dsize=(opt.resize_size, opt.resize_size))
    haze = Image.fromarray(haze)
    clear = Image.fromarray(clear)

    haze1 = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(haze)[None, ::].to(opt.device)
    clear1 = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(clear)[None, ::]
    haze_no = tfs.ToTensor()(haze)[None, ::]
    clear_no = tfs.ToTensor()(clear)[None, ::].to(opt.device)
    with torch.no_grad():
        pred, _ = net(haze1, clear_no)

    ssim1 = metrics.ssim(pred, clear_no).item()
    psnr1 = metrics.psnr(pred, clear_no)
    ssim2 = metrics.ssim(haze1, clear_no).item()
    psnr2 = metrics.psnr(haze1, clear_no)

    ssim_all.append(ssim1)
    psnr_all.append(psnr1)

print('ssim', np.average(np.array(ssim_all)), 'psnr', np.average(np.array(psnr_all)))
