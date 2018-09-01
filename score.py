from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image

import time
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path


PRINT_SECONDS = 0.2

session_count = 0
def trackSession():
    global session_count
    if session_count == 0:
        session_count += 1
        return True
    else:
        return False
    return False

def setSession(file, flag=True):
    now = datetime.now()
    if flag:
        file.write("\n-------------------- SESSION - {} -------------------------\n".format(now.strftime("%Y-%m-%d %H:%M")))
    else:
        file.write("-------------------- SESSION - {} -------------------------\n".format(now.strftime("%Y-%m-%d %H:%M")))

def endSession(flag=True):
    now = datetime.now()
    if flag:
        return "\n-------------------- END SESSION - {} -------------------------\n".format(now.strftime("%Y-%m-%d %H:%M"))
    else:
        return "-------------------- END SESSION - {} -------------------------\n".format(now.strftime("%Y-%m-%d %H:%M"))
    return ""

def delayPrint(string, seconds): # n seconds delay printing
    time.sleep(seconds)
    exportLogs(string)
    print(string)

def exportLogs(logs, f="demo/logs.log"):
    logs += "\n"
    if(isfile(f)):
        file = open(f, "a")
        if trackSession():
            setSession(file)
        file.write(logs)
        file.close()
    else:
        print("Log file does not exist!")
        print("Creating {} file...".format(f))
        file = open(f, "a+")
        if trackSession():
            setSession(file)
        file.write(logs)
        file.close()

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        delayPrint("Ground truth: {}".format(net.blobs[gt].data[0].flatten().shape), PRINT_SECONDS)
        delayPrint("Segmeted output: {}".format(net.blobs[layer].data[0].argmax(0).flatten().shape), PRINT_SECONDS)
        # fixing the bug of shape mismatch, ground truth has only the shape of X columns and not X * Y
        # hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
        #                         net.blobs[layer].data[0].argmax(0).flatten(),
        #                         n_cl)
        hist += fast_hist(net.blobs[gt].data[0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label'):
    # print '>>>', datetime.now(), 'Begin seg tests'
    delayPrint(">>>{} Begin seg tests".format(datetime.now()), PRINT_SECONDS)
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    # print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    delayPrint(">>>{} Iteration: {} Loss: {}".format(datetime.now(), iter, loss), PRINT_SECONDS)
    # overall accuracy
    # acc = np.diag(hist).sum() / hist.sum()
    over_acc = np.diag(hist).sum() / hist.sum()
    # print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    delayPrint(">>>{} Iteration: {} Overall accuracy: {}".format(datetime.now(), iter, over_acc), PRINT_SECONDS)
    # per-class accuracy
    # acc = np.diag(hist) / hist.sum(1)
    mean_acc = np.diag(hist) / hist.sum(1)
    # print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    delayPrint(">>>{} Iteration: {} Mean Accuracy: {}".format(datetime.now(), iter, np.nanmean(mean_acc)), PRINT_SECONDS)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    delayPrint(">>>{} Iteration: {} Mean IU: {}".format(datetime.now(), iter, np.nanmean(iu)), PRINT_SECONDS)
    freq = hist.sum(1) / hist.sum()
    # print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
    #         (freq[freq > 0] * iu[freq > 0]).sum()
    delayPrint(">>>{} Iteration: {} Fwavacc: {}".format(datetime.now(), iter, (freq[freq > 0] * iu[freq > 0]).sum()), PRINT_SECONDS)
    # return hist
    # returns a dictionary of results
    return {'loss' : loss, 'over_acc' : float(over_acc), 'mean_acc' : float(np.nanmean(mean_acc)), 'iu' : float(np.nanmean(iu)), 'freq' : float((freq[freq > 0] * iu[freq > 0]).sum())}
