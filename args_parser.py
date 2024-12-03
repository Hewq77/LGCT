
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='LGCT')
    parser.add_argument('-root', type=str, default='../Datasets/')
    parser.add_argument('-dataset', type=str, default='Houston',
                        choices=['PaviaU', 'Houston','Chikusei','YRE'])
    parser.add_argument('--scale_ratio', type=float, default=4)
    parser.add_argument('--n_bands', type=int, default=145) # PaviaU:103, Botswana:145#
    parser.add_argument('--n_select_bands', type=int, default=5)
    parser.add_argument('--model_path', type=str,
                        default='./checkpoints',
                        help='path for trained encoder')
    parser.add_argument('--pretrained_model_path', type=str,
                        default=None)
    # learning settingl
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help='end epoch for training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--criterion', type=str, default='L1',
                        help='Loss = L1 or l1ssim')
    # hyperparams(window size and group)
    parser.add_argument('--win_size', type=int, default=8, help='4,8,16,32')
    parser.add_argument('--group', type=int, default=32, help='4,8,16,32')

    args = parser.parse_args()
    return args
