import argparse
import logging
import os
import random

import torch
import torchvision
import wandb
from data.data_load import data_load
from models.AE import VAE
from models.aunet import AU_NET
from utils.metrics import get_acc, get_f1
from utils.utils import *
from utils.utils_callbacks import CallBackEvaluation, CallBackLogging
from utils.utils_logging import AverageMeter, init_logging

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='parameters for action unit recognition')
parser.add_argument('--batchsize', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--epochs', type=int, default=12, metavar='N', help='training epochs')
parser.add_argument('--num_workers', type=int, default=8, metavar='N', help='5,8')

parser.add_argument('--data', type=str, default='bp4d', choices=['bp4d', 'disfa'])
parser.add_argument('--subset', type=int, default=1, choices=[1, 2, 3])
parser.add_argument('--alpha', type=int, default=0.9)
parser.add_argument('--beta', type=int, default=0.1)

parser.add_argument('--vae', type=str, default='', help='vae model checkpoint')
parser.add_argument('--weight', type=float, default=1)

def main():
    args = parser.parse_args()

    manualSeed = random.randint(1, 10000)
    set_seed(manualSeed)

    cur_path = os.path.abspath(os.curdir)
    
    if args.data == 'bp4d':
        args.nclasses = 12
        au_net = VAE(12, 8)
    elif args.data == 'disfa':
        args.nclasses = 8
        au_net = VAE(8, 6)
    else:
        raise NotImplementedError()
    
    #save path
    args.output = f'./Results/{args.data}/{args.subset}'
    os.makedirs(args.output, exist_ok=True)

    # log
    log_root = logging.getLogger()
    init_logging(log_root, args.output)


    transform = transforms.Compose([transforms.ToTensor()])

    model = AU_NET(alpha=args.alpha, beta=args.beta, n_classes=args.nclasses)

    if args.data == 'disfa':
        print('do not forget to load pretrained bp4d weights')
    #load data
    train_data = data_load(data=args.data,phase='train', subset=args.subset, transform=transform, seed=manualSeed)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True,num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_data = data_load(data=args.data,phase='val', subset=args.subset, transform=transform, seed=manualSeed)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize, shuffle=True,num_workers=args.num_workers, pin_memory=True, drop_last=False)

    val_data = data_load(data=args.data,phase='val', subset=args.subset, transform=transform, flip=True,seed=manualSeed)
    val_loader_flip = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize, shuffle=True,num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print('train num:%d, val num:%d ' % (len(train_loader.dataset), len(val_loader.dataset)))

    # AU prediction model
    model = nn.DataParallel(model).cuda()
    params = list(model.parameters())
    sub_params = [p for p in params if p.requires_grad]
    print('num of params', sum(p.numel() for p in sub_params))
    optimizer = torch.optim.AdamW(sub_params, lr=0.0001, betas=(0.9, 0.999), weight_decay=0)

    #vae model
    au_net.load_state_dict(torch.load(args.path, 'cpu'))
    au_net = nn.DataParallel(au_net).cuda()
    au_net.eval()

    # loss
    au_weights = torch.from_numpy(train_data.AU_weight).float().cuda()
    criterion = nn.BCEWithLogitsLoss(weight=au_weights)


    total_step = int(len(train_data) / args.batchsize * args.epochs)
    callback_logging = CallBackLogging(len(train_loader) // 4, total_step, args.batchsize, None)
    callback_validation = CallBackEvaluation(val_loader, val_loader_flip, subset='valflip', multi=True)

    # training
    global_step = 0
    losses = AverageMeter()
    losses_kd = AverageMeter()
    acces = AverageMeter()
    f1es = AverageMeter()

    for epoch in range(args.epochs):
        model.train()

        #frozen fan
        model.module.feature.eval()

        for index, data in enumerate(train_loader):
            global_step += 1

            images_1 = data['img_former'].cuda()
            images_2 = data['img_later'].cuda()
            images = data['img'].cuda()
            label = data['label'].cuda()
            pred = model(images, images_1, images_2)
            loss = criterion(pred, label)

            # VAE loss
            pred_au, m_s, v_s = au_net(torch.sigmoid(pred))
            with torch.no_grad():
                gt_au, m_t, v_t = au_net(label.detach())
            loss_m = F.mse_loss(m_s, m_t.detach())
            loss_v = torch.mean(v_s)
            loss_kd = loss_m + loss_v
            loss_all = loss + loss_kd * args.weight

            if epoch == 0 and index == 1:
                torchvision.utils.save_image(images, '%s/mid.png' % args.output, normalize=True)
                torchvision.utils.save_image(images_1, '%s/former.png' % args.output, normalize=True)
                torchvision.utils.save_image(images_2, '%s/later.png' % args.output, normalize=True)

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            batch_size = images.size(0)
            pred = pred.detach().data.cpu().float()
            label = label.detach().data.cpu().float()

            f1_score = get_f1(label, pred)
            acc = get_acc(label, pred)

            losses.update(loss.detach().item(), 1)
            losses_kd.update(loss_kd.detach().item(), 1)
            acces.update(acc.mean().detach().item(), batch_size)
            f1es.update(f1_score.mean().detach().item(), batch_size)
            callback_logging(global_step, losses, losses_kd, acces, f1es, epoch, optimizer)


        callback_validation(epoch, model)
        torch.save(model.module.predictor.state_dict(), os.path.join(args.output, "fc_%d.pth" % epoch))
        lr_change(epoch + 1, optimizer)


if __name__ == '__main__':
    main()
