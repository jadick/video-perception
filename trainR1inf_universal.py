import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
#from utils import *
import matplotlib.pyplot as plt
import torch.nn.utils.spectral_norm as spectralnorm
from torch.nn import init
import torchvision
import torch
import torch.nn as nn
from models import *
from utils import *
import torch.nn.functional as F
from helper import *

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

import argparse

"""
dim = 128
z_dim = 2
lambda_gp = 50
bs =64
d_penalty = 0
skip_fq = 1
total_epochs = 200
lambda_P = 0
lambda_MSE = 10
"""
parser = argparse.ArgumentParser('training config')
parser.add_argument('--total_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--lambda_gp', type=int, default=10, help='number of epochs of training')
parser.add_argument('--bs', type=int, default=64, help='size of the batch')
parser.add_argument('--dim', type=int, default=128, help='common_dim')
parser.add_argument('--z_dim', type=int, default=1, help='z dim')
parser.add_argument('--L', type=int, default=2, help='z dim')
parser.add_argument('--skip_fq', type=int, default=5, help='loop frequency for WGAN')
parser.add_argument('--d_penalty', type=float, default=0.0, help='diversity penalty')
parser.add_argument('--lambda_P', type=float, default=0.0, help='Perceptual Penalty, keep at 1.0')
parser.add_argument('--lambda_PM', type=float, default=0.0, help='Perceptual Penalty Marginal, keep at 1.0')
parser.add_argument('--lambda_MSE', type=float, default=1.0, help='Perceptual Penalty')
parser.add_argument('--path', type=str, default='./data/', help='Data Path')
parser.add_argument('--pre_path', type=str, default='None', help='Pretrained_Path')
parser.add_argument('--single_bit', type=bool, default=False)

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
        grad_outputs=fake[:,0],
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def cal_W1(ssf, discriminator, discriminator_M, test_loader):
    mse_loss = nn.MSELoss(reduction='sum')

    set_models_state([ssf, discriminator, discriminator_M], 'eval')

    W1_distance = []
    W1M_distance = []
    MSE = []

    num_x = 0
    for i, x in enumerate(iter(test_loader)):
        with torch.no_grad():
            #Get the data
            x = x.permute(0, 4, 1, 2, 3)
            x = x.cuda().float()
            x_cur = x[:,:,1,...]
            x_ref = x[:,:,0,...]
            x_hat = ssf(x_cur, x_ref)


            fake_vid = torch.cat((x[:,:,0,:,:], x_hat), dim = 1).detach()
            real_vid = x[:,0,:2,...].detach() #this looks good!

            fake_validity = discriminator(fake_vid)
            real_validity = discriminator(real_vid)

            fake_img = x_hat.detach()
            real_img = x[:,0,6:7,...].detach()
            fake_valid_m = discriminator_M(fake_img)
            real_valid_m = discriminator_M(real_img)

            W1_distance.append(torch.sum(real_validity) - torch.sum(fake_validity))
            W1M_distance.append(torch.sum(real_valid_m) - torch.sum(fake_valid_m))
            #print (F.mse_loss(x[:,:,1,:,:], x_hat)* x.size()[0])
            MSE.append(mse_loss(x[:,:,1,:,:], x_hat))
            #print (mse_loss(x[:,:,1,:,:], x_hat)/(64*64*len(x)))
            num_x += len(x)

    W1_distance = torch.Tensor(W1_distance)
    W1M_distance = torch.Tensor(W1M_distance)
    MSE = torch.Tensor(MSE)

    return W1M_distance.sum()/num_x, W1_distance.sum()/num_x, MSE.sum()/(64*64*num_x)

def set_models_state(list_models, state):
    if state =='train':
        for model in list_models:
            model.train()
    else:
        for model in list_models:
            model.eval()

def set_opt_zero(opts):
    for opt in opts:
        opt.zero_grad()


def main():
    args = parser.parse_args()
    #Params
    dim = args.dim#128
    z_dim = args.z_dim #2
    lambda_gp = args.lambda_gp #50
    bs = args.bs #64
    d_penalty = args.d_penalty #0
    skip_fq = args.skip_fq #10
    total_epochs = args.total_epochs #200
    lambda_P = args.lambda_P
    lambda_PM = args.lambda_PM
    lambda_MSE = args.lambda_MSE
    L = args.L
    path = args.path
    pre_path = args.pre_path
    single_bit = args.single_bit

    #Create folder
    #Create folder:
    folder_name='universal_dim_'+str(dim)+'|z_dim_'+str(z_dim)+'|L_' + str(L)+'|_single_bit'+str(single_bit)+ +'|lambda_gp_'+str(lambda_gp) \
        +'|bs_'+str(bs)+'|dpenalty_'+str(d_penalty)+'|lambdaP_'+str(lambda_P)+'|lambdaPM_'+str(lambda_PM)+'|lambdaMSE_' + str(lambda_MSE)
    print ("Settings: ", folder_name)

    os.makedirs('./saved_models/'+ folder_name, exist_ok=True)
    f = open('./saved_models/'+ folder_name + "/performance.txt", "a")

    #Define Models
    discriminator = Discriminator_v3(out_ch=2) #Generator Side
    discriminator_M = Discriminator_v3(out_ch=1) #Marginal Discriminator
    ssf = ScaleSpaceFlow(num_levels=1, dim=z_dim, stochastic=True, quantize_latents=True, L=L, freeze_enc=True, single_bit=single_bit)

    list_models = [discriminator, discriminator_M, ssf]

    ssf.cuda()
    discriminator.cuda()
    discriminator_M.cuda()

    #Load models:
    if pre_path != 'None':
        #prefix_path = 'z'+str(z_dim)+'l'+str(L)+'_MMSE'
        ssf.motion_encoder.load_state_dict(torch.load(pre_path+'/m_enc.pth'))
        ssf.motion_decoder.load_state_dict(torch.load(pre_path+'/m_dec.pth'))
        ssf.P_encoder.load_state_dict(torch.load(pre_path+'/p_enc.pth'))
        ssf.res_encoder.load_state_dict(torch.load(pre_path+'/r_enc.pth'))
        ssf.res_decoder.load_state_dict(torch.load(pre_path+'/r_dec.pth'))
        discriminator.load_state_dict(torch.load(pre_path+'/discriminator.pth'))
        discriminator_M.load_state_dict(torch.load(pre_path+'/discriminator_M.pth'))
    else:
        print ('invalid!')
        exit()

    ssf.motion_encoder.eval()
    ssf.motion_decoder.eval()
    ssf.res_encoder.eval()

    #Define Data Loader
    train_loader, test_loader = get_dataloader(data_root=path, seq_len=8, batch_size=bs, num_digits=1)
    mse = torch.nn.MSELoss()

    #discriminator.train()
    opt_ssf= torch.optim.RMSprop(ssf.parameters(), lr=5e-5)
    opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=5e-5)
    opt_dm = torch.optim.RMSprop(discriminator_M.parameters(), lr=5e-5)

    list_opt = [opt_ssf, opt_d, opt_dm]

    ssf.motion_encoder.eval()
    ssf.res_encoder.eval()

    for param in ssf.motion_encoder.parameters():
        print(param.data[0])
        break

    for param in ssf.motion_decoder.parameters():
        print(param.data[0])
        break

    for param in ssf.res_encoder.parameters():
        print(param.data[0])
        break

    for epoch in range(total_epochs):
        set_models_state(list_models, 'train')
        for i,x in enumerate(iter(train_loader)):
            #Set 0 gradient
            set_opt_zero(list_opt)

            #Get the data
            x = x.permute(0, 4, 1, 2, 3)
            x = x.cuda().float()
            x_cur = x[:,:,1,...]
            x_ref = x[:,:,0,...]
            x_hat = ssf(x_cur, x_ref)


            #Optimize discriminator
            fake_vid = torch.cat((x[:,:,0,:,:], x_hat), dim = 1)
            real_vid = x[:,0,:2,...].detach() #this looks good!
            fake_validity = discriminator(fake_vid.detach())
            real_validity = discriminator(real_vid)
            gradient_penalty = compute_gradient_penalty(discriminator, real_vid.data, fake_vid.data)
            errVD =  -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            errVD.backward()
            opt_d.step()

            #Optimize discriminator M
            fake_img = x_hat.detach()
            real_img = x[:,0,:1,...].detach()
            fake_valid_m = discriminator_M(fake_img)
            real_valid_m = discriminator_M(real_img)
            gradient_penalty_m = compute_gradient_penalty(discriminator_M, fake_img.data, real_img.data)
            errID =  -torch.mean(real_valid_m) + torch.mean(fake_valid_m) + lambda_gp * gradient_penalty_m
            errID.backward()
            opt_dm.step()

            if i%skip_fq == 0:
                x_cur = x_cur.detach()
                x_ref = x_ref.detach()
                x_hat = ssf(x_cur, x_ref)

                fake_vid = torch.cat((x_ref, x_hat), dim = 1)
                fake_validity = discriminator(fake_vid)
                errVG = -torch.mean(fake_validity)

                fake_img = x_hat
                fake_validity_im = discriminator_M(fake_img)
                errIG = -torch.mean(fake_validity_im)

                loss = lambda_MSE*mse(x_hat, x_cur) + lambda_P*errVG +  lambda_PM*errIG
                loss.backward()

                opt_ssf.step()

        if epoch %10 == 0:
            show_str= "Epoch: "+ str(epoch) + "l_PM, l_P, l_MSE, d_penalty " + str(lambda_PM) + str(lambda_P)+ " " \
            +str(lambda_MSE) + " " + str(d_penalty) + " P loss: " + str(cal_W1(ssf, discriminator, discriminator_M, test_loader))
            print (show_str)

            f.write(show_str+"\n")

    show_str= "Epoch: "+ str(epoch) + "l_PM, l_P, l_MSE, d_penalty " + str(lambda_PM) + str(lambda_P)+ " " \
            +str(lambda_MSE) + " " + str(d_penalty) + " P loss: " + str(cal_W1(ssf, discriminator, discriminator_M, test_loader))
    print (show_str)
    f.write(show_str+"\n")

    set_models_state(list_models, 'eval')

    torch.save(ssf.motion_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_enc.pth'))
    torch.save(ssf.motion_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_dec.pth'))
    torch.save(ssf.P_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'p_enc.pth'))
    torch.save(ssf.res_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_enc.pth'))
    torch.save(ssf.res_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_dec.pth' ))
    torch.save(discriminator.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator.pth'))
    torch.save(discriminator_M.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_M.pth'))

    f.close()

    for param in ssf.motion_encoder.parameters():
        print(param.data[0])
        break

    for param in ssf.motion_decoder.parameters():
        print(param.data[0])
        break

    for param in ssf.res_encoder.parameters():
        print(param.data[0])
        break

    #save some figures
    for i,x in enumerate(iter(train_loader)):
        x = x.permute(0, 4, 1, 2, 3)
        x = x.cuda().float()
        break
    np.savez_compressed("./saved_models/" + folder_name+"/x", a=x.detach().cpu().numpy())

    for i in range(5): #generate same figure 5 times
        x_cur = x[:,:,1,...]
        x_ref = x[:,:,0,...]
        x_hat = ssf(x_cur, x_ref)
        np.savez_compressed("./saved_models/" + folder_name+"/x_hat"+str(i), a=x_hat.detach().cpu().numpy())

if __name__ == "__main__":
    main()
