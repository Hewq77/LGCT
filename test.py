import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.optim
from models.LGCT_arch import LGCT

from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, calc_cc, calc_ssim
from data_loader import build_datasets
import args_parser
import cv2
from time import *
from thop import profile

args = args_parser.args_parser()

print(args)

def process_image(image, red, green, blue):
    image = np.squeeze(image)
    red_channel = image[red, :, :][:, :, np.newaxis]
    green_channel = image[green, :, :][:, :, np.newaxis]
    blue_channel = image[blue, :, :][:, :, np.newaxis]
    image = np.concatenate((blue_channel, green_channel, red_channel), axis=2)
    image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def main():
    args.root = os.path.join(args.root + args.dataset)
    train_list, test_list = build_datasets(args.root,
                                           args.dataset,
                                           args.image_size,
                                           args.n_select_bands,
                                           args.scale_ratio)
    
    # Set the number of bands for each dataset
    dataset_bands = {
        'PaviaU': 103,
        'Houston': 144,
        'Chikusei': 128,
        'YRE': 280
    }
    if args.dataset in dataset_bands:
        args.n_bands = dataset_bands[args.dataset]

    if args.arch == 'LGCT':
        model = LGCT(img_size=args.image_size, upscale=args.scale_ratio,
                     in_chans1=args.n_select_bands + args.n_bands, in_chans2=args.n_bands,
                     embed_dim=48, dim_head=192, num_heads1=[8, 8, 8],
                     window_size=8, group=8, dim=48, num_heads2=[8, 8, 8], ffn_expansion_factor=2.66,
                     LayerNorm_type='WithBias', bias=False)

    # Load the trained model parameters
    model_path = os.path.join(args.model_path, args.arch, args.dataset)
    train_best_epoch = '2024_11_27_00_21_14/net_9946epoch.pth' # root path of pre-trained model weight
    model_path = os.path.join(model_path, train_best_epoch)
  
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print('Load the chekpoint of {}'.format(model_path)) #COLORMAP_PARULA

    test_ref, test_lr, test_hr = test_list
    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()

    begin_time = time()
    if args.arch == 'LGCT':
        out = model(lr, hr)
    end_time = time()
    run_time = (end_time - begin_time)

    print()
    print()
    flops, params = profile(model, inputs=(lr, hr))
    print('Dataset:   {}'.format(args.dataset))
    print('Arch:   {}'.format(args.arch))
    print('params(M),', params / 1e6)
    print('flops(G),', flops / 1e9)
    print('Test time(s),', run_time)
    print()

    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    cc = calc_cc(ref, out)
    ssim = calc_ssim(ref, out)

    print('RMSE:   {:.6f};'.format(rmse))
    print('PSNR:   {:.6f};'.format(psnr))
    print('ERGAS:   {:.6f};'.format(ergas))
    print('cc:   {:.6f};'.format(cc))
    print('SAM:   {:.6f};'.format(sam))
    print('SSIM:   {:.6f}.'.format(ssim))

    # bands order
    if args.dataset == 'PaviaU':
        red = 66
        green = 28
        blue = 0
    elif args.dataset == 'Houston':
        red = 60
        green = 29
        blue = 7
    elif args.dataset == 'Chikusei':
        red = 80
        green = 76
        blue = 2

    save_path = os.path.join('./figs', args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lr = process_image(lr.detach().cpu().numpy(), red, green, blue)
    lr = cv2.resize(lr, (out.shape[2], out.shape[3]), interpolation=cv2.INTER_NEAREST)
    out = process_image(out, red, green, blue)
    ref = process_image(ref, red, green, blue)

    cv2.imwrite('./figs/{}/{}_lr.jpg'.format(args.dataset, args.dataset), lr)
    cv2.imwrite('./figs/{}/{}_{}_out.png'.format(args.dataset, args.dataset, args.arch), out)
    cv2.imwrite('./figs/{}/{}_ref.png'.format(args.dataset, args.dataset), ref)

    lr_dif = np.uint8(5 * np.abs((lr - ref)))
    lr_dif = cv2.cvtColor(lr_dif, cv2.COLOR_BGR2GRAY)
    lr_dif = cv2.applyColorMap(lr_dif, cv2.COLORMAP_PARULA)
    cv2.imwrite('./figs/{}/{}_lr_dif.png'.format(args.dataset, args.dataset), lr_dif)

    out_dif = np.uint8(5 * np.abs((out - ref)))
    out_dif = cv2.cvtColor(out_dif, cv2.COLOR_BGR2GRAY)
    out_dif = cv2.applyColorMap(out_dif, cv2.COLORMAP_PARULA)
    cv2.imwrite('./figs/{}/{}_{}_out_dif.png'.format(args.dataset, args.dataset, args.arch), out_dif)

    print()
    print('Test achieve!')

if __name__ == '__main__':
    main()
