import torch
from torch import nn
from utils import to_var, batch_ids2words
import random
import torch.nn.functional as F
import cv2


def spatial_edge(x):
    edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

    return edge1, edge2

def spectral_edge(x):
    edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

    return edge


def train(train_list,
          image_size,
          scale_ratio,
          n_bands,
          arch,
          model,
          optimizer,
          criterion,
          epoch,
          n_epochs):
    train_ref, train_lr, train_hr = train_list

    h, w = train_ref.size(2), train_ref.size(3)
    h_str = random.randint(0, h-image_size-1)
    w_str = random.randint(0, w-image_size-1)

    train_ref = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]

    train_ref_numpy = train_ref.permute(0, 2, 3, 1).squeeze().numpy()
    train_lr = cv2.GaussianBlur(train_ref_numpy, (7, 7), 2)
    train_lr = cv2.resize(train_lr, (train_lr.shape[1] // scale_ratio, train_lr.shape[0] // scale_ratio))
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0)

    train_hr = train_hr[:, :, h_str:h_str+image_size, w_str:w_str+image_size]

    model.train()

    # Set mini-batch dataset
    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()

    # Forward, Backward and Optimize
    optimizer.zero_grad()

    out = model(image_lr, image_hr)
    loss = criterion(out, image_ref)

    loss.backward()
    optimizer.step()
    # scheduler.step()

    # Print log info
    print('Epoch [%d/%d], Loss: %.4f'
          %(epoch,
            n_epochs,
            loss,
            )
         )
