import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import torch
from torch import nn, optim
from models.LGCT_arch import LGCT

from utils import *
from data_loader import build_datasets
from validate import validate
from train import train
import datetime
import args_parser
from loss_utils import *

args = args_parser.args_parser()
print (args)

def setup_model():
    # Custom dataloader
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

    # Build the model
    if args.arch == 'LGCT':
        model = LGCT(img_size=args.image_size, upscale=args.scale_ratio,
                     in_chans1=args.n_select_bands + args.n_bands, in_chans2=args.n_bands,
                     embed_dim=48, dim_head=192, num_heads1=[8, 8, 8],
                     window_size=8, group=8, dim=48, num_heads2=[8, 8, 8], ffn_expansion_factor=2.66,
                     LayerNorm_type='WithBias', bias=False).cuda()

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")

    if args.criterion == 'L1':
        criterion = nn.L1Loss().cuda()
    elif args.criterion == 'l1ssim':
        criterion = HybridL1SSIM(channel=8, weighted_r=(1.0, 0.1))

    return model, optimizer, train_list, test_list, criterion

def setup_logging():
    date_time = time2file_name(str(datetime.datetime.now()))
    model_path = os.path.join(args.model_path, args.arch, args.dataset, date_time)
    os.makedirs(model_path, exist_ok=True)

    log_dir = os.path.join(model_path, 'train.log')
    logger = initialize_logger(log_dir)

    return model_path, logger

def main():
    # Setup model, optimizer, and data
    model, optimizer, train_list, test_list, criterion = setup_model()

    # Setup logging and model path
    model_path, logger = setup_logging()

    # Resume 
    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        print(f"Loaded checkpoint: {args.pretrained_model_path}")
        recent_rmse, recent_psnr, recent_ergas, recent_sam = validate(test_list, args.arch, model, 0, args.n_epochs)
        print('psnr: ', recent_psnr)      

    # Validation
    _, best_psnr, _, _ = validate(test_list, args.arch, model, 0, args.n_epochs)
    print ('psnr: ', best_psnr)

    # Start training and validation
    print ('Start Training: ')
    best_epoch = 0
    for epoch in range(args.n_epochs):
        # One epoch's training
        print ('Train_Epoch_{}: '.format(epoch))
        train(train_list,
              args.image_size,
              args.scale_ratio,
              args.n_bands,
              args.arch,
              model,
              optimizer,
              criterion,
              epoch,
              args.n_epochs)

        # 20 epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_rmse, recent_psnr, recent_ergas, recent_sam = validate(test_list,
                                args.arch,
                                model,
                                epoch,
                                args.n_epochs)

        print('rmse: %f, psnr: %f, ergas: %f, sam: %f' %
                  (recent_rmse, recent_psnr, recent_ergas, recent_sam))
        if epoch % 20 == 0:
            logger.info('Epoch [%d/%d], rmse: %f, psnr: %f, ergas: %f, sam: %f, best_psnr: %f' %
                    (epoch, args.n_epochs, recent_rmse, recent_psnr, recent_ergas, recent_sam, best_psnr))
        # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
           best_epoch = epoch
           if epoch >= 5000:
              torch.save(model.state_dict(), os.path.join(model_path, 'net_%depoch.pth' % epoch))
              print ('Saved!')
              print ('')

        print('best psnr:', best_psnr, 'at epoch:', best_epoch)

    print(f"Training Complete. Best PSNR: {best_psnr:.4f}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time() - start_time
    print(f"Training Time is {end_time:.2f} seconds")
