'''
code is modified from https://torchbearer.readthedocs.io/en/0.2.1/examples/vae.html
'''
from __future__ import print_function

import argparse
import logging
import os

import numpy as np
import torch
import torch.utils.data
from datasets.label_loader import label_load
from models.AE import VAE
from torch import nn, optim
from torch.nn import functional as F
from utils.evaluation import evaluate_metrics
from utils.metrics import get_acc, get_f1
from utils.utils_logging import init_logging

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='vae on AU labels')
parser.add_argument('--batchsize', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--data', type=str, default='bp4d', choices=['bp4d', 'disfa'])
parser.add_argument('--subset', type=int, default=1, choices=[1, 2, 3])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--model', type=str,default='vae',choices=['vae'])
parser.add_argument('--weight', type=float, default=0.1,choices=[0.1,0.2,0.3])
args = parser.parse_args()

args.cuda = True
args.num_workers = 8
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

# default: 8 for bp4d, 6 for disfa, 5 for aw
if args.data in ['bp4d']:
    args.label_dim = 12
    args.dim = 8
elif args.data=='disfa':
    args.label_dim = 8
    args.dim = 6
else:
    raise NotImplementedError()


args.output = f'./results/{args.data}_{args.subset}'
os.makedirs(args.output, exist_ok=True)
log_root = logging.getLogger()
init_logging(log_root, args.output)


print('loading train set')
train_data = label_load(data_name=args.data, phase='train', subset=args.subset, seed=args.seed)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
args.log_interval = len(train_loader)//4

print('loading val set')
val_data = label_load(data_name=args.data, phase='test', subset=args.subset)
test_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)

'''
model the distribution of multi labels
we introduce a regularization term during the optimization by explicitly constraining N (µi,σi) 
to be close to a normal distribution, N (0, I),
measured by Kullback-Leibler divergence (KLD) between
these two distributions. 
'''


model = VAE(label_dim=args.label_dim, dim=args.dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.wd)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu=None, logvar=None):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    if mu is None:
        return BCE
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE,KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,label) in enumerate(train_loader):
        data = data.to(device)
        label = label.to(device)
        recon_batch, mu, logvar = model(data)
        loss_bce, loss_kl = loss_function(recon_batch, label, mu, logvar)
        loss = loss_bce+loss_kl*args.weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            if args.model == 'ae':
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBCE: {:.6f} \tKL: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_bce.item() / len(data),loss_kl.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    if epoch == args.epochs:
        torch.save(model.state_dict(), os.path.join(args.output, "vae_%d.pth" % epoch))

def test(epoch):
    model.eval()
    num = 0
    for i, (data,label) in enumerate(test_loader):
        num = num+data.size(0)

        data = data.to(device)
        with torch.no_grad():
            recon_batch, mu, logvar = model(data,is_train=True)
            pred = recon_batch.data.cpu().float()
        loss_bce, loss_kl = loss_function(recon_batch.cpu().data, label, mu, logvar)
        if i:
            preds = torch.cat((preds, pred), dim=0)
            gts = torch.cat((gts, label), dim=0)
            loss_bces = loss_bces+loss_bce.item()
            loss_kls = loss_kls+loss_kl.item()
        else:
            preds = pred
            gts = label
            loss_bces = loss_bce.item()
            loss_kls = loss_kl.item()

    results = evaluate_metrics(gts,preds,{'ACC': get_acc, 'F1': get_f1})
    f1 = np.mean(results['F1'].numpy())
    acc = np.mean(results['ACC'].numpy())
    bce = loss_bces/float(num)
    kl = loss_kls/float(num)
    logging.info('[%d]F1-Current:%.3f Accuracy-Current:%.3f BCE-Loss:%.3f KL-Loss:%.3f' %(epoch,f1,acc,bce,kl))



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
