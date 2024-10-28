import argparse
import os
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader
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
import time

cuda = True
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

import argparse
# test

parser = argparse.ArgumentParser('training config')
parser.add_argument('--total_epochs', type = int, default = 100, help = 'Number of training epochs')
parser.add_argument('--lambda_gp', type = int, default = 10, help = 'Gradient penalty')
parser.add_argument('--bs', type = int, default = 64, help = 'batch size')
parser.add_argument('--dim', type = int, default = 128, help = 'Common dimension')
parser.add_argument('--eps_1', type = int, default = 2, help = 'second frame bitrate')
parser.add_argument('--eps_2', type = int, default = 16, help = 'third frame bitrate')
parser.add_argument('--L', type = int, default = 2, help = 'Quantization parameter')
parser.add_argument('--skip_fq', type = int, default=5, help = 'Loop frequency for WGAN')
parser.add_argument('--d_penalty', type = float, default = 0.0, help = 'Diversity penalty')
parser.add_argument('--lambda_JD', type = float, default = 0.0, help = 'Joint perceptual penalty')
parser.add_argument('--lambda_MSE', type = float, default = 1.0, help = 'MSE Penalty')
parser.add_argument('--lambda_NEW', type = float, default = 0.0, help = 'New perceptual penalty')
parser.add_argument('--lambda_FMD', type = float, default = 0.0, help = 'Marginal perceptual penalty')
parser.add_argument('--path', type = str, default = './data/', help = 'Data path')
parser.add_argument('--pre_path', type = str, default = './fixed_models/', help ='Path to pretrained weights')
parser.add_argument('--pre_path_fixed', type = str, default = './fixed_models/', help ='Path to the fixed model')
parser.add_argument('--single_bit', type=int, default=0)
parser.add_argument('--step', type=int, default=15, help ='step size for mmnist')
parser.add_argument('--pre_trained', type = str, default = 'JD', help ='fixed model used to pretrain for mse')
parser.add_argument('--pre_trained_lambda', type = float, default = 0, help ='lambda for fixed model used to pretrain for mse')
parser.add_argument('--dataset', type = str, default = 'mmnist_4_axis_random_sample_step', help ='mmnist variant to use')
parser.add_argument('--learning_rate', type=int, default= 1, help ='learning rate for ssf model(1e-5)')


def set_models_state(list_models, state, FMD, JD, NEW):
    if state =='train':
        for model in list_models:
            if model != None:
                model.train()
    else:
        for model in list_models:
            if model != None:
                model.eval()

def set_opt_zero(opts, FMD, JD, NEW):
    for opt in opts:
        if opt != None:
            opt.zero_grad()

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
    #gradients = gradients.view(gradients.size(0), -1)
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def cal_W1(ssf1,ssf2, discriminator_JD, discriminator_NEW, discriminator_FMD, test_loader, list_models, FMD, JD, NEW):
    mse_loss = nn.MSELoss(reduction = 'sum')
    mse_avg = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_models_state(list_models, 'eval', FMD, JD, NEW)

    JD_distance = [] if JD else None
    NEW_distance = [] if NEW else None
    FMD_distance = [] if FMD else None
    MSE = []

    num_x = 0
    for i, x in enumerate(iter(test_loader)):
        with torch.no_grad():
            x = x.permute(0, 4, 1, 2, 3).to(device).float()
            x0 = x[:,:,0,...]
            x1 = x[:,:,1,...]
            x2 = x[:,:,2,...]
            x1_hat = ssf1(x1, x0)
            x2_hat = ssf2(x2, x1_hat, x0, x1_hat)

            if JD:
                fake_vid_JD = torch.cat((x0, x1_hat, x2_hat), dim = 1).detach()
                real_vid_JD = torch.cat((x0, x1, x2), dim = 1).detach() 
                fake_validity_JD = discriminator_JD(fake_vid_JD)
                real_validity_JD = discriminator_JD(real_vid_JD)
                
            if NEW:
                fake_vid_NEW = torch.cat((x0, x1_hat, x2_hat), dim = 1).detach()
                real_vid_NEW = torch.cat((x0, x1_hat, x2), dim = 1).detach()
                fake_validity_NEW = discriminator_NEW(fake_vid_NEW)
                real_validity_NEW = discriminator_NEW(real_vid_NEW)
                
            if FMD:
                fake_img = x2_hat.detach()
                real_img = x2.detach()
                fake_validity_FMD = discriminator_FMD(fake_img)
                real_validity_FMD = discriminator_FMD(real_img)
                
            if JD:
                JD_distance.append(torch.sum(real_validity_JD) - torch.sum(fake_validity_JD))
            if NEW:
                NEW_distance.append(torch.sum(real_validity_NEW) - torch.sum(fake_validity_NEW))
            if FMD:
                FMD_distance.append(torch.sum(real_validity_FMD) - torch.sum(fake_validity_FMD))
                
            MSE.append(mse_loss(x2, x2_hat))
            num_x += len(x)

    JD_distance = torch.Tensor(JD_distance).sum() / num_x if JD else torch.tensor([0])
    NEW_distance = torch.Tensor(NEW_distance).sum() / num_x if NEW else torch.tensor([0])
    FMD_distance = torch.Tensor(FMD_distance).sum() / num_x if FMD else torch.tensor([0])
    MSE = torch.Tensor(MSE).sum() / (64*64*num_x)
    
    return JD_distance, MSE, NEW_distance, FMD_distance

def main():
    start = time.time()
    args = parser.parse_args()
    eps1 = args.eps_1
    eps2 = args.eps_2
    learning_rate = args.learning_rate
    zdim_1 = eps1//2
    zdim_2= eps2//2
    dim = args.dim
    dataset = dim = args.dataset
    lambda_gp = args.lambda_gp
    bs = args.bs
    d_penalty = args.d_penalty
    skip_fq = args.skip_fq 
    total_epochs = args.total_epochs
    lambda_JD = args.lambda_JD * 1e-3
    lambda_NEW = args.lambda_NEW * 1e-3
    lambda_FMD = args.lambda_FMD * 1e-3
    lambda_MSE = args.lambda_MSE
    pre_trained_lambda = args.pre_trained_lambda
    L = args.L
    path = args.path
    pre_path = args.pre_path
    pre_trained = args.pre_trained
    if 'unidir' in dataset:
        step = [11,5]
    else:
        step= args.step
    single_bit = bool(args.single_bit)
    pre_path_fixed = args.pre_path_fixed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #set perceptual flags
    if (lambda_FMD == 0 and lambda_JD == 0 and lambda_NEW == 0): 
        FMD = JD = NEW = True
        folder_name = f'inf-eps1-eps2/step_{step}/inf-{eps1}-{eps2}|lambdaJD_{lambda_JD}|'\
        + f'lambdaFMD_{lambda_FMD}|lambdaNEW_{lambda_NEW}'\
        + f'|lambdaMSE_{lambda_MSE}|pretrained_{pre_trained}_{pre_trained_lambda}|dataset_{dataset}'
        print ("Settings: ", folder_name)
    else:
        FMD = lambda_FMD > 0
        JD = lambda_JD > 0
        NEW = lambda_NEW > 0
        folder_name = f'inf-eps1-eps2/step_{step}/inf-{eps1}-{eps2}|lambdaJD_{lambda_JD}|'\
        + f'lambdaFMD_{lambda_FMD}|lambdaNEW_{lambda_NEW}|lambdaMSE_{lambda_MSE}|dataset_{dataset}'
        print ("Settings: ", folder_name)
        
    #set stoch/quant:
    stochastic = True
    quantize_latents = True
    if L == -1:
        stochastic = False
        quantize_latents = False
    
    #Create folder:
    

    os.makedirs('./saved_models/'+ folder_name, exist_ok=True)
    f = open('./saved_models/'+ folder_name + "/performance.txt", "a")
    print_str = f'l_NEW: {lambda_NEW}, l_JD : {lambda_JD}, l_FMD: {lambda_FMD},'\
                + f' l_MSE: {lambda_MSE},step: {step}, d_penalty: {d_penalty}, L: {L}, z_dim: {zdim_1}-{zdim_2}, bs: {bs}, dataset: {dataset}\n'
    f.write(print_str)
    f.write('| EPOCH |      PLF-JD     |      PLF-FMD    |      PLF-NEW    |      MSE        |\n')
    f.close()

    #Define Models
    discriminator_JD = Discriminator_v3(out_ch=3).to(device) if JD else None 
    discriminator_NEW = Discriminator_v3(out_ch=3).to(device) if NEW else None 
    discriminator_FMD = Discriminator_v3(out_ch=1).to(device) if FMD else None
    ssf1 = ScaleSpaceFlow(num_levels=1,dim=zdim_1,stochastic=True, quantize_latents=True,L=L,single_bit=single_bit).to(device)
    ssf2 =  ScaleSpaceFlow_R1eps_e2e_3frames(num_levels=1, dim=zdim_2, stochastic=True, quantize_latents=True, L=L, single_bit=single_bit,num_c=1, activation=torch.sigmoid).to(device)
    list_models = [discriminator_JD, discriminator_NEW, discriminator_FMD, ssf1, ssf2] 
    #load fixed model weights
    ssf1.motion_encoder.load_state_dict(torch.load(pre_path_fixed + '/m_enc.pth'))
    ssf1.motion_decoder.load_state_dict(torch.load(pre_path_fixed + '/m_dec.pth'))
    ssf1.P_encoder.load_state_dict(torch.load(pre_path_fixed + '/p_enc.pth'))
    ssf1.res_encoder.load_state_dict(torch.load(pre_path_fixed + '/r_enc.pth'))
    ssf1.res_decoder.load_state_dict(torch.load(pre_path_fixed + '/r_dec.pth'))
    #Load models:
    if pre_path != 'None':
        print(f'Initializing weights from: {pre_path}')
        ssf2.motion_encoder.load_state_dict(torch.load(pre_path + '/m_enc.pth'))
        ssf2.motion_decoder.load_state_dict(torch.load(pre_path + '/m_dec.pth'))
        ssf2.P_encoder.load_state_dict(torch.load(pre_path + '/p_enc.pth'))
        ssf2.res_encoder.load_state_dict(torch.load(pre_path + '/r_enc.pth'))
        ssf2.res_decoder.load_state_dict(torch.load(pre_path + '/r_dec.pth'))
        if JD:
            discriminator_JD.load_state_dict(torch.load(pre_path + '/discriminator_JD.pth'))
        if NEW:
            discriminator_NEW.load_state_dict(torch.load(pre_path + '/discriminator_NEW.pth'))
        if FMD:
            discriminator_FMD.load_state_dict(torch.load(pre_path + '/discriminator_FMD.pth'))
  

    #Define Data Loader
    train_loader, test_loader = get_dataloader(dataset=dataset,data_root=path, seq_len=3, batch_size=bs, num_digits=1,step=step)
    mse = torch.nn.MSELoss()

    #Define optimizers
    opt_ssf1= torch.optim.RMSprop(ssf1.parameters(), lr=learning_rate * 1e-5)
    opt_ssf2= torch.optim.RMSprop(ssf2.parameters(), lr=learning_rate * 1e-5)
    opt_JD = torch.optim.RMSprop(discriminator_JD.parameters(), lr =5e-5) if JD else None
    opt_NEW = torch.optim.RMSprop(discriminator_NEW.parameters(), lr=5e-5) if NEW else None
    opt_FMD = torch.optim.RMSprop(discriminator_FMD.parameters(), lr=5e-5) if FMD else None
    list_opt = [opt_ssf1, opt_ssf2, opt_JD, opt_NEW, opt_FMD] 

    for epoch in range(total_epochs):
        #print(epoch,lambda_NEW)
        a = time.time()
        set_models_state(list_models, 'train', FMD, JD, NEW)
        for i,x in enumerate(iter(train_loader)):
            #Set 0 gradient
            set_opt_zero(list_opt, FMD, JD, NEW)
            #Get the data
            x = x.permute(0, 4, 1, 2, 3).to(device).float()
            
            ##############  3-FRAME VIDEO (X0-X1-X2) ############################
            x0 = x[:,:,0,...]
            x1 = x[:,:,1,...]
            x2 = x[:,:,2,...]
            with torch.no_grad():
                x1_hat = ssf1(x1, x0)
                x2_hat = ssf2(x2, x1_hat, x0, x1_hat) #ssf(x_cur,x_ref)

            # optimize discriminator_JD
            if JD:
                fake_vid_JD = torch.cat((x0, x1_hat, x2_hat), dim = 1).detach()
                real_vid_JD = torch.cat((x0, x1, x2), dim = 1).detach() 
                fake_validity_JD = discriminator_JD(fake_vid_JD)
                real_validity_JD = discriminator_JD(real_vid_JD)
                gp_JD = compute_gradient_penalty(discriminator_JD, real_vid_JD.data, fake_vid_JD.data)
                errJD =  -torch.mean(real_validity_JD) + torch.mean(fake_validity_JD) + lambda_gp * gp_JD
                errJD.backward()
                opt_JD.step()

            # optimize discriminator_NEW
            if NEW:
                fake_vid_NEW = torch.cat((x0, x1_hat, x2_hat), dim = 1).detach()
                real_vid_NEW = torch.cat((x0, x1_hat, x2), dim = 1).detach()
                fake_validity_NEW = discriminator_NEW(fake_vid_NEW)
                real_validity_NEW = discriminator_NEW(real_vid_NEW)
                gp_NEW = compute_gradient_penalty(discriminator_NEW, real_vid_NEW.data, fake_vid_NEW.data)
                errNEW =  -torch.mean(real_validity_NEW) + torch.mean(fake_validity_NEW) + lambda_gp * gp_NEW
                errNEW.backward()
                opt_NEW.step()

            # optimize discriminator_FMD
            if FMD:
                fake_img = x2_hat.detach()
                real_img = x2.detach()
                fake_validity_FMD = discriminator_FMD(fake_img)
                real_validity_FMD = discriminator_FMD(real_img)
                gp_FMD = compute_gradient_penalty(discriminator_FMD, fake_img.data, real_img.data)
                errFMD =  -torch.mean(real_validity_FMD) + torch.mean(fake_validity_FMD) + lambda_gp * gp_FMD
                errFMD.backward()
                opt_FMD.step()
            

            if i%skip_fq == 0:
                x0 = x0.detach()
                x1 = x1.detach()
                x2 = x2.detach()  
                x1_hat = x1_hat.detach()
                x2_hat = ssf2(x2, x1_hat, x0, x1_hat)
                
                if JD:
                    fake_vid_JD = torch.cat((x0, x1_hat, x2_hat), dim = 1)
                    fake_validity_JD = discriminator_JD(fake_vid_JD)
                    errJD_ = -torch.mean(fake_validity_JD)
                else:
                    errJD_ = 0
                    
                if NEW:
                    fake_vid_NEW = torch.cat((x0, x1_hat, x2_hat), dim = 1)
                    fake_validity_NEW = discriminator_NEW(fake_vid_NEW)
                    errNEW_ = -torch.mean(fake_validity_NEW)
                else:
                    errNEW_ = 0
                    
                if FMD:
                    fake_img = x2_hat
                    fake_validity_FMD = discriminator_FMD(fake_img)
                    errFMD_ = -torch.mean(fake_validity_FMD)
                else:
                    errFMD_ = 0

                loss2 = lambda_MSE*mse(x2_hat, x2) + lambda_JD*errJD_ + lambda_NEW*errNEW_ + lambda_FMD*errFMD_
                loss2.backward()
                opt_ssf2.step()
        ############## ########################## ############################
        
        if ((epoch+1) % 10 == 0) or epoch == 0:
            f = open('./saved_models/'+ folder_name + "/performance.txt", "a")
            JD_loss, MSE_loss, NEW_loss, FMD_loss = cal_W1(ssf1,ssf2, discriminator_JD, discriminator_NEW, discriminator_FMD, test_loader, list_models, FMD, JD, NEW)
            show_str = f'   {epoch+1}   {JD_loss.item()} {FMD_loss.item()} {NEW_loss.item()} {MSE_loss.item()}'
            f.write(show_str + "\n")
            f.close()
            print(print_str)
            print(show_str)
            set_models_state(list_models, 'eval', FMD, JD, NEW)
            #print('saving at:',os.path.join("./saved_models/" + folder_name, 'm_enc.pth'))
            torch.save(ssf2.motion_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_enc.pth'))
            torch.save(ssf2.motion_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_dec.pth'))
            torch.save(ssf2.P_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'p_enc.pth'))
            torch.save(ssf2.res_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_enc.pth'))
            torch.save(ssf2.res_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_dec.pth' ))
            if JD:
                torch.save(discriminator_JD.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_JD.pth'))
            if NEW:
                torch.save(discriminator_NEW.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_NEW.pth'))
            if FMD:
                torch.save(discriminator_FMD.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_FMD.pth'))
    
    end = time.time()
    set_models_state(list_models, 'eval', FMD, JD, NEW)
    torch.save(ssf2.motion_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_enc.pth'))
    torch.save(ssf2.motion_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_dec.pth'))
    torch.save(ssf2.P_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'p_enc.pth'))
    torch.save(ssf2.res_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_enc.pth'))
    torch.save(ssf2.res_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_dec.pth' ))
    if JD:
        torch.save(discriminator_JD.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_JD.pth'))
    if NEW:
        torch.save(discriminator_NEW.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_NEW.pth'))
    if FMD:
        torch.save(discriminator_FMD.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_FMD.pth'))
    f = open('./saved_models/'+ folder_name + "/performance.txt", "a")
    f.write(f'Total Training Time: {(end-start)/3600} hours \n')
    f.close()

if __name__ == "__main__":
    main()
