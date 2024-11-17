import os, argparse
import torch, warnings

warnings.filterwarnings('ignore')
datasets = 'OHAZE'  # 'ITS' 'DENSE' 'NH' '6K' 'OTS' 'OHAZE'
type = 'main'
bs = 1
data_number = 0
epoch = 0
sleep_epoch = 0
eval_epoch = 0
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)

parser.add_argument('--datasets', type=str, default=datasets)
parser.add_argument('--blocks', type=int, default=1, help='residual_blocks')

parser.add_argument('--steps', type=int, default=data_number * epoch // bs)
parser.add_argument('--data_number', type=int, default=data_number // bs)
parser.add_argument('--eval_step', type=int, default=data_number * eval_epoch // bs)
parser.add_argument('--resize_size', type=int, default=240,
                    help='Takes effect when using --resize ')
parser.add_argument('--bs', type=int, default=bs, help='batch size')
parser.add_argument('--model_dir', type=str, default='./trained_models/')

parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--resume', type=bool, default=True)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--trainset', type=str, default='its_train')
parser.add_argument('--testset', type=str, default='its_test')
parser.add_argument('--net', type=str, default='Group_DRDC')
parser.add_argument('--gps', type=int, default=3, help='residual_groups')
parser.add_argument('--sleep_freq', type=int, default=data_number * sleep_epoch // bs)
parser.add_argument('--sleep_time', type=int, default=300)
parser.add_argument('--resize', default=True, action='store_true')
parser.add_argument('--no_lr_sche', default='cos', action='store_true', help='no lr cos schedule')
parser.add_argument('--perloss', action='store_true', default=True, help='perceptual loss')

# test
parser.add_argument('--task', type=str, default='its', help='its or ots')
parser.add_argument('--test_imgs', type=str, default='test/hazy', help='Test imgs folder')
parser.add_argument('--clear_imgs', type=str, default='test/clear/', help='Test imgs folder')

opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = opt.trainset + '_' + opt.net.split('.')[0] + '_' + str(opt.gps) + '_' + str(
    opt.blocks) + '_' + opt.datasets
opt.model_dir_last = opt.model_dir + model_name + '_last' + '.pk'
opt.model_dir = opt.model_dir + model_name + '_best' + '.pk'
log_dir = 'logs/' + model_name

print(opt)
print('model_dir:', opt.model_dir)

if not os.path.exists('trained_models'):
    os.mkdir('trained_models')
if not os.path.exists('numpy_files'):
    os.mkdir('numpy_files')
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('samples'):
    os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
    os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
