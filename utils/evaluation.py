import sys
import time

import numpy as np
import torch


def evaluate_metrics(ground_truth,predictions,metrics):
    results = {}
    for name, metric in metrics.items():
        results[name] = metric(ground_truth,predictions)
    return results

def evaluate_flip(net, dataloader_no_flip, dataloader_flip, metrics=None, sel_index=None):
    net.eval()
    # Loop without flip
    start_time=time.time()
    for index, data in enumerate(dataloader_no_flip):
        if index%100==0:
            end_time = time.time()
            sys.stdout.write('\r %d/%d %.2f' % (index, len(dataloader_no_flip),end_time-start_time))
            sys.stdout.flush()
            start_time = time.time()
        images = data['img'].cuda()
        label = data['label']
        with torch.no_grad():
            pred = net(images)
        pred = pred.data.cpu().float()
        if index:
            preds = torch.cat((preds, pred), dim=0)
            gts = torch.cat((gts, label), dim=0)
        else:
            preds = pred
            gts = label
    if dataloader_flip is not None:
        # Loop with flip
        n_images = 0
        for index, data in enumerate(dataloader_flip):
            if index % 100 == 0:
                sys.stdout.write('\r %d/%d' % (index, len(dataloader_no_flip)))
                sys.stdout.flush()
            images = data['img'].cuda()
            with torch.no_grad():
                pred = net(images)
            pred = pred.data.cpu().float()
            for k in range(0, images.size(0)):
                preds[n_images] = (pred[k] + preds[n_images]) / 2.0
                n_images += 1
    if sel_index is not None:
        preds = preds[:,sel_index]
    results = evaluate_metrics(gts,preds,metrics=metrics)
    net.train()
    return results


def evaluate_multi(net, dataloader_no_flip,dataloader_flip,metrics=None, sel_index=None):
    net.eval()
    start_time=time.time()
    for index, data in enumerate(dataloader_no_flip):
        if index%100==0:
            end_time = time.time()
            sys.stdout.write('\r %d/%d %.2f' % (index, len(dataloader_no_flip),end_time-start_time))
            sys.stdout.flush()
            start_time = time.time()

        images_1 = data['img_former'].cuda()
        images = data['img'].cuda()
        images_2 = data['img_later'].cuda()

        label = data['label']
        with torch.no_grad():
            pred = net(images,images_1,images_2)
        pred = pred.data.cpu().float()
        if index:
            preds = torch.cat((preds, pred), dim=0)
            gts = torch.cat((gts, label), dim=0)
        else:
            preds = pred
            gts = label

    if dataloader_flip is not None:
        # Loop with flip
        n_images = 0
        for index, data in enumerate(dataloader_flip):
            if index % 100 == 0:
                sys.stdout.write('\r %d/%d' % (index, len(dataloader_flip)))
                sys.stdout.flush()
            images_1 = data['img_former'].cuda()
            images = data['img'].cuda()
            images_2 = data['img_later'].cuda()
            with torch.no_grad():
                pred = net(images, images_1, images_2)
            pred = pred.data.cpu().float()
            for k in range(0, images.size(0)):
                preds[n_images] = (pred[k] + preds[n_images]) / 2.0
                n_images += 1


    if sel_index is not None:
        preds = preds[:,sel_index]
    results = evaluate_metrics(gts,preds,metrics=metrics)
    net.train()
    return results


