import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

import torch.nn.utils.spectral_norm as spectralnorm
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
#from models import *
from utils import *
from helper import *
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available else cpu)
from vimeo90k import Vimeo90kDataset, VideoFolder_diffusion

import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
device = torch.device('cuda' if torch.cuda.is_available else cpu)
from models import Discriminator_v3
from ssf_model import ScaleSpaceFlow
import time

LAMBDA_GP = 50
LAMBDA_MSE = 1
EPOCHS = 50
LAMBDA_AR = 0.08

model_name = f'AR_{LAMBDA_AR}'

def load_ssf_model(model, pre_path):
    model.motion_encoder.load_state_dict(torch.load(pre_path+'/m_enc.pth'))
    model.motion_decoder.load_state_dict(torch.load(pre_path+'/m_dec.pth'))
    model.P_encoder.load_state_dict(torch.load(pre_path+'/p_enc.pth'))
    model.res_encoder.load_state_dict(torch.load(pre_path+'/r_enc.pth'))
    model.res_decoder.load_state_dict(torch.load(pre_path+'/r_dec.pth'))
    return model

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0),  1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
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

train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop(256)])

train_dataset = VideoFolder_diffusion(
        "./data/vimeo-90k/vimeo_triplet/",
        rnd_interval=False,
        rnd_temp_order=False,
        split="train",
        transform=train_transforms)

train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True)

discriminator_AR = Discriminator_v3(ch=256, out_ch=6).to(device)
discriminator_AR.load_state_dict(torch.load('./saved_models/vimeo-90k/mse/discriminator_AR.pth'))
ssf = ScaleSpaceFlow().to(device)
ssf = load_ssf_model(ssf, './saved_models/vimeo-90k/mse/')
opt_AR = torch.optim.Adam(discriminator_AR.parameters(), lr=1e-5)
opt_ssf = torch.optim.Adam(ssf.parameters(), lr=5e-5)
mse = torch.nn.MSELoss()

a = time.time()
for epoch in range(EPOCHS):
    mse_list = []
    AR_list = []
    mse_epoch = 0
    AR_epoch = 0
    ssf.train()
    discriminator_AR.train()
    for i, data in enumerate(train_dataloader):
        if i%100 == 0:
            print(f'batch {i} of {len(train_dataloader)}')
        opt_AR.zero_grad()
        opt_ssf.zero_grad()

        x1 = 2*(data[:,0,...]-0.5)
        x2 = 2*(data[:,1,...]-0.5)
        x1_hat = 2*(data[:,3,...]-0.5)
        x1=x1.to(device)
        x2=x2.to(device)
        x1_hat = x1_hat.to(device)

        with torch.no_grad():
            x2_hat = ssf([x1_hat, x2])

        #change code
        real_vid = torch.cat((x1_hat, x2), dim = 1)
        fake_vid = torch.cat((x1_hat, x2_hat), dim =1)
        fake_validity = discriminator_AR(fake_vid.detach())
        real_validity = discriminator_AR(real_vid)

        #gradient_penalty = compute_gradient_penalty(discriminator, real_vid.data, fake_vid.data)
        errAR =  -torch.mean(real_validity) + torch.mean(fake_validity) #+ LAMBDA_GP * gradient_penalty
        errAR.backward()
        opt_AR.step()
        AR_list.append(errAR.item())

        # ssf optim
        x1 = x1.detach()
        x2 = x2.detach()
        x1_hat = x1_hat.detach()
        x2_hat = ssf([x1_hat, x2])
        fake_vid = torch.cat((x1_hat, x2_hat), dim =1)
        fake_validity = discriminator_AR(fake_vid.detach())

        errAR_ = -torch.mean(fake_validity)
        x2_hat = ssf([x1_hat, x2])
        loss = LAMBDA_MSE * mse(x2_hat, x2) + LAMBDA_AR*errAR_

        mse_list.append(loss.item())
        loss.backward()
        opt_ssf.step()

    if epoch % 1 == 0:
        mse_epoch = torch.Tensor(mse_list).mean().item()
    AR_epoch = torch.Tensor(AR_list).mean().item()

    b = time.time()
    run_time = (b-a)/60
    print(f'| EPOCH: {epoch} | MSE LOSS: {mse_epoch} | AR LOSS: {AR_epoch} | TIME: {run_time} min|')

    if epoch % 1 == 0:
        print('saving models...')
        ssf.eval()
        discriminator_AR.eval()
        torch.save(discriminator_AR.state_dict(), f"./saved_models/vimeo-90k/{model_name}/discriminator_AR.pth")
        torch.save(ssf.motion_encoder.state_dict(), f'./saved_models/vimeo-90k/{model_name}/m_enc.pth')
        torch.save(ssf.motion_decoder.state_dict(), f'./saved_models/vimeo-90k/{model_name}/m_dec.pth')
        torch.save(ssf.P_encoder.state_dict(), f'./saved_models/vimeo-90k/{model_name}/p_enc.pth')
        torch.save(ssf.res_encoder.state_dict(), f'./saved_models/vimeo-90k/{model_name}/r_enc.pth')
        torch.save(ssf.res_decoder.state_dict(), f'./saved_models/vimeo-90k/{model_name}/r_dec.pth')
