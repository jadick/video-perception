import argparse
import math
import random
import shutil

import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './hfilc/')
from hfilc.compress import prepare_model, prepare_dataloader, compress_and_save, load_and_decompress, compress_and_decompress

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import VideoFolder, VideoFolder_diffusion
from compressai.optimizers import net_aux_optimizer#
from compressai.zoo import video_models#

from matplotlib import pyplot as plt
from Discriminator import Discriminator
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser('training config')
parser.add_argument('--lambda_P', type=float, default=0.03, help='Perceptual Penalty, keep at 1.0')


args = parser.parse_args()


total_epoch=50
lambda_gp = 50
lambda_P = args.lambda_P

print ("whut?", lambda_P)

def hwc_tonp(x):
    #x = (x+1)/2.0
    #x = x.clamp(0.0, 1.0)
    x = x.detach().cpu().numpy()
    x = x.transpose([0,2,3,1])
    return x


# In[2]:


import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0),  1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    #print (fake.shape)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake[:],
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# In[3]:


train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(256)]
    )

train_dataset = VideoFolder_diffusion(
        "../../vimeo_triplet/",
        rnd_interval=False,
        rnd_temp_order=False,
        split="train",
        transform=train_transforms,
    )

train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )


# In[4]:


from ssf_model import ScaleSpaceFlow


# In[5]:


ssf = ScaleSpaceFlow()
#ssf.train()


# In[6]:


ssf.load_state_dict(torch.load("./ckpt_ssf/ssf_GAN_old/ssf_GAN_19tensor(0.0220, device=\'cuda:0\', grad_fn=<MseLossBackward0>)0.008.pth"))
ssf.cuda()

discriminator = Discriminator((6, 256, 256))
discriminator.cuda()


# In[8]:


mse = torch.nn.MSELoss()
#BCE = torch.nn.functional.binary_cross_entropy_with_logits()
ssf_opt = torch.optim.Adam(ssf.parameters(), lr=5e-5)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=5e-5)

ssf = nn.DataParallel(ssf)
discriminator = nn.DataParallel(discriminator)

print (len(train_dataloader))

for epoch in range(100):
    for i, data in enumerate(train_dataloader):
        ssf_opt.zero_grad()
        opt_d.zero_grad()

        #with torch.no_grad():
        x1 = 2*(data[0]-0.5)
        x2 = 2*(data[1]-0.5)
        x1_hat = 2*(data[3] -0.5)

        x1=x1.cuda()
        x2=x2.cuda()
        x1_hat = x1_hat.cuda()

        x2_hat = ssf([x1_hat, x2])

        real_vid = torch.cat((x1, x2), dim = 1)
        fake_vid = torch.cat((x1_hat, x2_hat), dim =1)

        fake_validity = discriminator(fake_vid.detach())#[0]
        #fake_validity = fake_validity[1]
        real_validity = discriminator(real_vid)#[0]
        #real_validity = real_validity[1]
        #gradient_penalty = compute_gradient_penalty(discriminator, real_vid.data, fake_vid.data)
        errVD =  F.binary_cross_entropy_with_logits(input=real_validity, target=torch.ones_like(real_validity)) + F.binary_cross_entropy_with_logits(input=fake_validity, target=torch.zeros_like(fake_validity))         #-torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        errVD.backward()
        opt_d.step()
        #print ("WGAN: ", errVD)
        if i % 2 == 0:
            #Optimize the encoder-decoder
            x1 = x1.detach()
            x2 = x2.detach()
            x1_hat = x1_hat.detach()
            x2_hat = ssf([x1_hat, x2])
            mse_loss = mse(x2_hat, x2)

            fake_vid = torch.cat((x1_hat, x2_hat), dim = 1)
            fake_validity = discriminator(fake_vid)
            errVG = F.binary_cross_entropy_with_logits(input=fake_validity, target=torch.ones_like(fake_validity))

            loss = mse_loss + lambda_P*errVG
            #
            loss.backward()
            ssf_opt.step()

        #break
    
    if epoch % 5 == 0:
        torch.save(ssf.module.state_dict(), os.path.join("./ckpt_ssf/ssf_GAN_" + str(epoch) + str(mse_loss) + str(lambda_P) + ".pth"))
        torch.save(discriminator.module.state_dict(), os.path.join("./ckpt_ssf/disc_GAN_"+ str(epoch) + str(errVD) + str(lambda_P) + ".pth"))
        print (mse_loss)
        print (errVD)
        

torch.save(ssf.module.state_dict(), os.path.join("./ckpt_ssf/ssf_GAN_" + str(epoch) + str(mse_loss) + str(lambda_P) + ".pth"))
torch.save(discriminator.module.state_dict(), os.path.join("./ckpt_ssf/disc_GAN_"+ str(epoch) + str(errVD) + str(lambda_P) + ".pth"))
