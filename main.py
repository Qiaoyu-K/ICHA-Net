from models import *
import time
from torch import optim
import warnings
from torch import nn

warnings.filterwarnings('ignore')
from option import model_name, log_dir
from data_utils import *
from torchvision.models import vgg19

print('log_dir:', log_dir)
print('model_name:', model_name)
print('dataset_dir:', opt.path)

models_ = {
    'Group_DRDC': Network(dim=3, kernel_size=3, blocks=opt.blocks),
}
loaders_ = {
    'its_train': ITS_train_loader,
    'its_test': ITS_test_loader,
}
start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion, opt, scheduler):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir, map_location=opt.device)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = int(ckp['step'])
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']

        print(f'start_step:{start_step} start training ---')
    else:
        print('train from scratch *** ')

    if not opt.no_lr_sche:
        pass
    else:
        scheduler.step(start_step)
    losses_L1 = []
    losses_dis = []

    for step in range(start_step + 1, opt.steps + 1):
        net.train()
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]

        x, y = next(iter(loader_train))
        x = x.to(opt.device)
        y = y.to(opt.device)
        out, dis_loss = net(x, y)
        loss_L1 = criterion[0](out, y)
        loss = loss_L1 + 0.1 * dis_loss
        if opt.perloss:
            loss2 = criterion[1](out, y)
            loss = loss + 0.04 * loss2

        loss.backward()

        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        losses_L1.append(loss_L1.cpu().item())
        losses_dis.append(dis_loss.cpu().item())
        psnr_train = psnr(out.detach(), y.detach())
        print(
            f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}'
            f' |psnr :{psnr_train}',
            end='', flush=True)
        if opt.sleep_freq is not None:
            if (step + 1) % opt.sleep_freq == 0:
                print(f'\nsleep for {opt.sleep_time}s')
                time.sleep(opt.sleep_time)
        if step % opt.eval_step == 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

            print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            losses_L1_mean = np.mean(losses_L1)
            losses_dis_mean = np.mean(np.array(losses_dis))
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'losses_L1': losses_L1,
                    'losses_dis': losses_dis,
                    'losses_L1_means': losses_L1_mean,
                    'losses_dis_means': losses_dis_mean,
                    'batch_size': opt.bs,
                    'model': net.state_dict()
                }, opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')
            else:
                torch.save({
                    'step': step,
                    'max_psnr': psnr_eval,
                    'max_ssim': ssim_eval,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'losses_L1': losses_L1,
                    'losses_dis': losses_dis,
                    'losses_L1_means': losses_L1_mean,
                    'losses_dis_means': losses_dis_mean,
                    'batch_size': opt.bs,
                    'model': net.state_dict()
                }, opt.model_dir_last)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


def test(net, loader_test):
    net.eval()
    with torch.cuda.device(opt.device):
        torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred, dis_loss = net(inputs, targets)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
    return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    if opt.perloss:
        vgg_model = vgg19(pretrained=True).features
        vgg_model = vgg_model.to(opt.device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        criterion.append(PerLoss(vgg_model).to(opt.device))
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10 * opt.data_number, gamma=0.95)  # 0.35
    optimizer.zero_grad()
    train(net, loader_train, loader_test, optimizer, criterion, opt, scheduler)
