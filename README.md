# LGCT: Local–Global Collaborative Transformer for Fusion of Hyperspectral and Multispectral Images (TGRS 2024)

Official code repo for [LGCT: Local–Global Collaborative Transformer for Fusion of Hyperspectral and Multispectral Images](https://ieeexplore.ieee.org/document/10742406) (TGRS'24)  [[PDF]](https://github.com/Hewq77/LGCT/blob/main/Others/LGCT_pdf)

Wangquan He, Xiyou Fu, Nanying Li, Qi Ren and Sen Jia.
## Network
<figure>
<img src=./Others/LGCT.png> 
<figcaption align = "center"><b> </b></figcaption>
</figure>

## Requirements

To install dependencies:

```setup
# create new anaconda env
conda create -n LGCT python=3.8 -y
conda activate LGCT

# install python dependencies
pip install -r requirements.txt
```

## Usage:
Before training, you need to:

- Download  Datasets : [Pavia University](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) / [Houston](https://hyperspectral.ee.uh.edu/?page_id=459) / [Chikusei](https://naotoyokoya.com/Download.html).

- Set the data path `-root` in ***args_parser.py***.

  
### 1. Training 
Use the following command to train the network. Take training LGCT in Houston dataset as an example,
```
 CUDA_VISIBLE_DEVICES=0 python -u python.py  \
    -arch 'LGCT' \
    -root '[root path of datasets]' \
    -dataset 'Houston' \
    --model_path './checkpoints'\
    --n_epochs 10000 --lr 1e-4\
    --criterion 'L1' \
```
### 2. Testing 
 Before testing, set the pre-trained model weight 'pth' files to line 57 in ***test.py***. These files are obtained through the training phase and can be found in the `./checkpoints`.
 
Then run the following command:
```
python test.py
```

## Citation
If you find this work helpful, please consider citing it. We would greatly appreciate it!
```
@article{he2024lgct,
  title={LGCT: Local-Global Collaborative Transformer for Fusion of Hyperspectral and Multispectral Images},
  author={He, Wangquan and Fu, Xiyou and Li, Nanying and Ren, Qi and Jia, Sen},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  publisher={IEEE}
}
```
## Acknowledgment
Some codes are brought from [BasicSR](https://github.com/XPixelGroup/BasicSR), [SSRNET](https://github.com/hw2hwei/SSRNET) and [BDT](https://github.com/shangqideng/BDT). We sincerely appreciate their contributions.
