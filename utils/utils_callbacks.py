import logging
import os
import time
from typing import List

import numpy as np
import torch
from utils.evaluation import evaluate_flip, evaluate_multi
from utils.metrics import get_acc, get_f1
from utils.utils_logging import AverageMeter


class CallBackEvaluation(object):
    def __init__(self,test_dataloader_no_flip, test_dataloader_flip=None, subset='valid', sel_index=None,multi=False):
        self.metrics = {'ACC': get_acc, 'F1': get_f1}
        self.test_dataloader_no_flip = test_dataloader_no_flip
        self.test_dataloader_flip = test_dataloader_flip
        self.highest_acc: float = 0.0
        self.best_accepoch: int = 0
        self.highest_f1: float = 0.0
        self.best_f1epoch: int = 0
        self.subset = subset
        self.sel_index = sel_index #np.array([0, 1, 2, 3, 6])
        self.multi = multi

    def __call__(self, epoch,model):
        results = {}
        if self.multi:
            output = evaluate_multi(model, self.test_dataloader_no_flip,self.test_dataloader_flip, metrics=self.metrics, sel_index=self.sel_index)
        else:
            output = evaluate_flip(model, self.test_dataloader_no_flip, self.test_dataloader_flip, metrics=self.metrics, sel_index=self.sel_index)

        output['ACC'] = output['ACC'].numpy()
        results['ACC'] = np.mean(output['ACC'])
        logging.info('%s current ACC'%(self.subset))
        if results['ACC'] > self.highest_acc:
            self.highest_acc = results['ACC']
            self.best_accepoch = epoch
        logging.info('%s [%d]Accuracy-Highest: %.5f [%d] Accuracy-Current:%.3f' %
                     (self.subset, self.best_accepoch, self.highest_acc, epoch, results['ACC']))
        outline = ','.join(["%.3f" % float(x) for x in output['ACC']])
        logging.info('acc score: '+outline)
        logging.info('%s current F1' % (self.subset))
        output['F1'] = output['F1'].numpy()
        results['F1'] = np.mean(output['F1'])
        if results['F1'] > self.highest_f1:
            self.highest_f1 = results['F1']
            self.best_f1epoch = epoch
        logging.info('%s [%d]F1score-Highest: %.5f [%d] F1score-Current:%.3f' %
                     (self.subset, self.best_f1epoch, self.highest_f1, epoch, results['F1']))
        outline = ','.join(["%.3f" % float(x) for x in output['F1']])
        logging.info('f1 score: '+ outline)
        return results



class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, writer=None):
        self.frequent: int = frequent
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.writer = writer
        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss, loss_kd, acc, f1, epoch, opt):
        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed
                except ZeroDivisionError:
                    speed = float('inf')
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if loss_kd is None:
                    msg = "Speed %.2f samples/sec Loss %.4f acc %.4f f1 %.4f Epoch: %d Global Step: %d LR1: %.8f Time: %.3f hours" % (
                        speed_total, loss.avg, acc.avg, f1.avg, epoch, global_step, opt.param_groups[0]['lr'], time_for_end)
                    logging.info(msg)
                else:
                    msg = "Speed %.2f samples/sec Loss %.4f kd %.4f acc %.4f f1 %.4f Epoch: %d Global Step: %d LR1: %.8f Time: %.3f hours" % (
                        speed_total, loss.avg, loss_kd.avg, acc.avg, f1.avg, epoch, global_step, opt.param_groups[0]['lr'], time_for_end)
                    logging.info(msg)
                    loss_kd.reset()

                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self,output="./"):
        self.output: str = output

    def __call__(self, epoch, backbone: torch.nn.Module):
        torch.save(backbone.module.state_dict(), os.path.join(self.output, "backbone_%d.pth"%epoch))

