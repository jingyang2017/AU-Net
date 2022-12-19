import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score


def get_recall(y_labels,outputs):
    outputs = torch.sigmoid(outputs)
    outputs_i = outputs + 0.5
    outputs_i = outputs_i.type(torch.int32)
    y_ilab = y_labels.type(torch.int32)
    temp=[]
    for i in range(y_ilab.size(1)):
        temp.append(recall_score(y_ilab[:,i], outputs_i[:,i], average='binary'))
    temp = np.array(temp,dtype=np.float)
    return temp

def get_f1(y_labels,outputs):
    # f1_ = F1(y_labels,outputs, F1_Thresh=0.5)
    outputs = torch.sigmoid(outputs)
    outputs_i = outputs + 0.5
    outputs_i = outputs_i.type(torch.int32)
    y_ilab = y_labels.type(torch.int32)
    # used in f1
    gd_num = torch.sum(y_ilab, dim=0)
    pr_num = torch.sum(outputs_i, dim=0)

    sum_ones = y_ilab + outputs_i
    pr_rtm = sum_ones // 2

    pr_rt = torch.sum(pr_rtm, dim=0)

    # prevent nan to destroy the f1
    pr_rt = pr_rt.type(torch.float32)
    gd_num = gd_num.type(torch.float32)
    pr_num = pr_num.type(torch.float32)

    zero_scale = torch.zeros_like(torch.min(pr_rt))

    if torch.eq(zero_scale, torch.min(gd_num)):
        gd_num += 1
    if torch.eq(zero_scale, torch.min(pr_num)):
        pr_num += 1
    if torch.eq(zero_scale, torch.min(pr_rt)):
        pr_rt += 0.01

    recall = pr_rt / gd_num
    precision = pr_rt / pr_num
    f1 = 2 * recall * precision / (recall + precision)
    return f1

# def get_f1(y_labels,outputs,F1_Thresh=0.5):
#     """Evaluates the IF1_SCORE
#     """
#     pred = torch.sigmoid(outputs)
#     gt = y_labels.numpy()
#     pred = pred.numpy()
#     gt = gt.transpose((1, 0))
#     pred = pred.transpose((1, 0))
#     preds = np.zeros(pred.shape)
#     preds[pred < F1_Thresh] = 0
#     preds[pred >= F1_Thresh] = 1
#     F1=[]
#     for ii in range(gt.shape[0]):
#         output = preds[ii]
#         gt_ = gt[ii]
#         temp = f1_score(gt_,output)
#         F1.append(temp)
#     F1 = np.array(F1,dtype=np.float)
#     F1 = torch.from_numpy(F1)
#     F1 = F1.type(torch.float32)
#     return F1


def get_acc(y_labels,outputs):
    outputs = torch.sigmoid(outputs)
    outputs_i = outputs + 0.5
    outputs_i = outputs_i.type(torch.int32)
    y_ilab = y_labels.type(torch.int32)
    # used in acc
    pr_rtm = torch.eq(outputs_i, y_ilab)
    pr_rt = torch.sum(pr_rtm, dim=0)
    pr_rt = pr_rt.type(torch.float32)
    acc = pr_rt / outputs.shape[0]
    return acc



def ACC(ground_truth,predictions):
    """Evaluates the mean accuracy
    """
    return np.mean(ground_truth.astype(int) == predictions.astype(int))



def ICC(labels,predictions):
    """Evaluates the ICC(3, 1)
    """
    naus = predictions.shape[1]
    icc = np.zeros(naus)
    n = predictions.shape[0]
    for i in range(0,naus):
        a = np.asmatrix(labels[:,i]).transpose()
        b = np.asmatrix(predictions[:,i]).transpose()
        dat = np.hstack((a, b))
        mpt = np.mean(dat, axis=1)
        mpr = np.mean(dat, axis=0)
        tm  = np.mean(mpt, axis=0)
        BSS = np.sum(np.square(mpt-tm))*2
        BMS = BSS/(n-1)
        RSS = np.sum(np.square(mpr-tm))*n
        tmp = np.square(dat - np.hstack((mpt,mpt)))
        WSS = np.sum(np.sum(tmp, axis=1))
        ESS = WSS - RSS
        EMS = ESS/(n-1)
        icc[i] = (BMS - EMS)/(BMS + EMS)
    return icc

def MAE(ground_truth,predictions):
    """Evaluates the Mean Absolute Error.
    """
    return np.mean(np.abs(ground_truth-predictions))