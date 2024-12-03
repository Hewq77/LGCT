import os
import numpy as np
import json
import torch
import h5py
from cv2 import imread, resize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import torch
import torch.nn as nn
from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence
import logging



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda().float()
    return Variable(x, volatile=volatile)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def batch_ids2words(batch_ids, vocab):

    batch_words = []
    for i in range(batch_ids.size(0)):
        sampled_caption = []
        ids = batch_ids[i,::].cpu().data.numpy()

        for j in range(len(ids)):
            id = ids[j]
            word = vocab.idx2word[id]
            # if word == '.':
            #     print ('.: ', id)
            if word == '<end>':
                break
            if '<start>' not in word:
                sampled_caption.append(word)

        for k in sampled_caption:
            if  k==sampled_caption[0]:
                sentence = k     
            else:
                sentence = sentence + ' ' + k

        sentence = u'{}'.format(sentence)   if sampled_caption!=[]  else  u'.'
        batch_words.append(sentence)

    return batch_words

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))