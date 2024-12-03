import torch
import numpy as np
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F
import sewar as sewar_api
from skimage.metrics import structural_similarity
import cv2


def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/4*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    img_max = np.max(img_tgt)
    psnr = 10*np.log10(img_max**2/mse)

    return psnr

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2, axis=0))
    B = np.sqrt(np.sum(img_fus**2, axis=0))
    AB = np.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B)
    sam = np.arccos(sam)
    sam = np.mean(sam)*180/3.1415926535

    return sam

def calc_cc_cuda(H_ref, H_fuse):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(H_ref_reshaped, 1).unsqueeze(1)

    CC = torch.sum((H_fuse_reshaped- mean_fuse)*(H_ref_reshaped-mean_ref), 1)/torch.sqrt(torch.sum((H_fuse_reshaped- mean_fuse)**2, 1)*torch.sum((H_ref_reshaped-mean_ref)**2, 1))

    CC = torch.mean(CC)
    return CC

def calc_cc(H_ref, H_fuse):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.reshape(N_spectral, -1)
    H_ref_reshaped = H_ref.reshape(N_spectral, -1)

    # Calculating mean value
    mean_fuse = np.mean(H_fuse_reshaped, 1)[:, np.newaxis]
    mean_ref = np.mean(H_ref_reshaped, 1)[:, np.newaxis]

    CC = np.sum((H_fuse_reshaped- mean_fuse)*(H_ref_reshaped-mean_ref), 1)/np.sqrt(np.sum((H_fuse_reshaped- mean_fuse)**2, 1)*np.sum((H_ref_reshaped-mean_ref)**2, 1))

    CC = np.mean(CC)
    return CC

def calc_ssim(img_tgt, img_fus, win_size=11 ):
    '''
    :param reference:
    :param target:
    :return:
    '''

    img_tgt = np.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = np.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    # img_tgt = img_tgt.cpu().numpy()
    # img_fus = img_fus.cpu().numpy()

    ssim = structural_similarity(img_tgt, img_fus,win_size=win_size)

    return ssim