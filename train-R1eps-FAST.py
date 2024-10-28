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
parser.add_argument('--z_dim', type = int, default = 1, help = 'Z dimension')
parser.add_argument('--L', type = int, default = 2, help = 'Quantization parameter')
parser.add_argument('--skip_fq', type = int, default=5, help = 'Loop frequency for WGAN')
parser.add_argument('--d_penalty', type = float, default = 0.0, help = 'Diversity penalty')
parser.add_argument('--lambda_JD', type = float, default = 0.0, help = 'Joint perceptual penalty')
parser.add_argument('--lambda_MSE', type = float, default = 1.0, help = 'MSE Penalty')
parser.add_argument('--lambda_NEW', type = float, default = 0.0, help = 'New perceptual penalty')
parser.add_argument('--lambda_FMD', type = float, default = 0.0, help = 'Marginal perceptual penalty')
parser.add_argument('--path', type = str, default = './data/', help = 'Data path')
parser.add_argument('--pre_path', type = str, default = './fixed_models/', help ='Path to pretrained weights')

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
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def cal_W1(ssf, encoder, decoder, decoder_hat, discriminator_JD, discriminator_NEW, discriminator_FMD, test_loader, list_models, FMD, JD, NEW):
    mse_loss = nn.MSELoss(reduction = 'sum')
    mse_avg = nn.MSELoss()
    set_models_state(list_models, 'eval', FMD, JD, NEW)

    JD_distance = [] if JD else None
    NEW_distance = [] if NEW else None
    FMD_distance = [] if FMD else None
    MSE = []

    num_x = 0
    for i, x in enumerate(iter(test_loader)):
        with torch.no_grad():
            #Get the data
            x = x.permute(0, 4, 1, 2, 3)
            x = x.cuda().float()
            x_cur = x[:,:,1,...]
            with torch.no_grad():
                hx = encoder(x[:,:,0,...])[0]
                x_ref = decoder(hx).detach()
                x_1_hat = decoder_hat(hx).detach()
            x_hat = ssf(x_cur, x_ref, x_1_hat)

            if JD:
                fake_vid = torch.cat((x_1_hat, x_hat), dim = 1).detach()
                real_vid = x[:,0,:2,...].detach() 
                fake_validity = discriminator_JD(fake_vid)
                real_validity = discriminator_JD(real_vid)
                
            if NEW:
                fake_vid = torch.cat((x_1_hat, x_hat), dim = 1).detach()
                real_vid_new = torch.cat((x_1_hat, x_cur), dim = 1).detach()
                new_metric_real_validity = discriminator_NEW(real_vid_new)
                new_metric_fake_validity = discriminator_NEW(fake_vid)
                
            if FMD:
                fake_img = x_hat.detach()
                real_img = x[:,0,6:7,...].detach()
                fake_valid_m = discriminator_FMD(fake_img)
                real_valid_m = discriminator_FMD(real_img)
                
            if JD:
                JD_distance.append(torch.sum(real_validity) - torch.sum(fake_validity))
            if NEW:
                NEW_distance.append(torch.sum(new_metric_real_validity) - torch.sum(new_metric_fake_validity))
            if FMD:
                FMD_distance.append(torch.sum(real_valid_m) - torch.sum(fake_valid_m))
                
            MSE.append(mse_loss(x[:,:,1,:,:], x_hat))
            num_x += len(x)

    JD_distance = torch.Tensor(JD_distance).sum() / num_x if JD else torch.tensor([0])
    NEW_distance = torch.Tensor(NEW_distance).sum() / num_x if NEW else torch.tensor([0])
    FMD_distance = torch.Tensor(FMD_distance).sum() / num_x if FMD else torch.tensor([0])
    MSE = torch.Tensor(MSE).sum() / (64*64*num_x)
    
    return JD_distance, MSE, NEW_distance, FMD_distance

def main():
    start = time.time()
    args = parser.parse_args()
    dim = args.dim
    z_dim = args.z_dim
    lambda_gp = args.lambda_gp
    bs = args.bs
    d_penalty = args.d_penalty
    skip_fq = args.skip_fq 
    total_epochs = args.total_epochs
    lambda_JD = args.lambda_JD * 1e-3
    lambda_NEW = args.lambda_NEW * 1e-3
    lambda_FMD = args.lambda_FMD * 1e-3
    lambda_MSE = args.lambda_MSE
    L = args.L
    path = args.path
    pre_path = args.pre_path
    bit_rate = (z_dim * 2 * np.log2(L)).astype(int)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # we check what is metric used for training (FMD,JD,NEW,MSE)
    # if only one flag is true, we train only the discriminator for the chosen metric
    if (lambda_FMD == 0 and lambda_JD == 0 and lambda_NEW == 0):
        FMD = JD = NEW = True
    else:
        FMD = lambda_FMD > 0
        JD = lambda_JD > 0
        NEW = lambda_NEW > 0

    #No quantization:
    stochastic = True
    quantize_latents = True
    if L == -1:
        stochastic = False
        quantize_latents = False
    
    #Create folder:
    folder_name = f'{bit_rate}/R1-eps|dim_{dim}|z_dim_{z_dim}|L_{L}|lambdaJD_{lambda_JD}|'\
    + f'lambdaFMD_{lambda_FMD}|lambdaNEW_{lambda_NEW}|lambdaMSE_{lambda_MSE}'
    print ("Settings: ", folder_name)

    os.makedirs('./saved_models/'+ folder_name, exist_ok=True)
    f = open('./saved_models/'+ folder_name + "/performance.txt", "a")
    print_str = f'l_NEW: {lambda_NEW}, l_JD : {lambda_JD}, l_FMD: {lambda_FMD},'\
                + f' l_MSE: {lambda_MSE}, d_penalty: {d_penalty}, L: {L}, z_dim: {z_dim}, bs: {bs}\n'
    f.write(print_str)
    f.write('| EPOCH |      PLF-JD     |      PLF-FMD    |      PLF-NEW    |      MSE        |\n')
    f.close()

    #Define Models
    discriminator_JD = Discriminator_v3(out_ch=2).to(device) if JD else None
    discriminator_NEW = Discriminator_v3(out_ch=2).to(device) if NEW else None
    discriminator_FMD = Discriminator_v3(out_ch=1).to(device) if FMD else None
    ssf = ScaleSpaceFlow_R1eps(num_levels=1, dim=z_dim, stochastic=stochastic, quantize_latents=quantize_latents, L=L).to(device)
    list_models = [discriminator_JD, discriminator_NEW, discriminator_FMD, ssf] 

    #Load models:
    if pre_path != 'None':
        print(f'Initializing weights from: {pre_path}')
        ssf.motion_encoder.load_state_dict(torch.load(pre_path + '/m_enc.pth'))
        ssf.motion_decoder.load_state_dict(torch.load(pre_path + '/m_dec.pth'))
        ssf.P_encoder.load_state_dict(torch.load(pre_path + '/p_enc.pth'))
        ssf.res_encoder.load_state_dict(torch.load(pre_path + '/r_enc.pth'))
        ssf.res_decoder.load_state_dict(torch.load(pre_path + '/r_dec.pth'))
        if JD:
            discriminator_JD.load_state_dict(torch.load(pre_path + '/discriminator_JD.pth'))
        if NEW:
            discriminator_NEW.load_state_dict(torch.load(pre_path + '/discriminator_NEW.pth'))
        if FMD:
            discriminator_FMD.load_state_dict(torch.load(pre_path + '/discriminator_FMD.pth'))
            
    #Define fixed model
    I_dim = 12 
    I_L = 2
    encoder = Encoder(dim=I_dim, nc=1, stochastic=True, quantize_latents=True, L=I_L).to(device).eval()
    decoder = Decoder_Iframe(dim=I_dim).to(device).eval()
    decoder_hat = Decoder_Iframe(dim=I_dim).to(device).eval()

    encoder.load_state_dict(torch.load('./I3/I_frame_encoder_zdim_12_L_2.pth'))
    decoder.load_state_dict(torch.load('./I3/I_frame_decoderMMSE_zdim_12_L_2.pth'))
    decoder_hat.load_state_dict(torch.load('./I3/I_frame_decoder_zdim_12_L_2.pth'))

    #Define Data Loader
    train_loader, test_loader = get_dataloader(data_root='./data/', seq_len=8, batch_size=bs, num_digits=1)
    mse = torch.nn.MSELoss()

    #Define optimizers
    opt_ssf= torch.optim.RMSprop(ssf.parameters(), lr=1e-5)
    opt_JD = torch.optim.RMSprop(discriminator_JD.parameters(), lr =5e-5) if JD else None
    opt_NEW = torch.optim.RMSprop(discriminator_NEW.parameters(), lr=5e-5) if NEW else None
    opt_FMD = torch.optim.RMSprop(discriminator_FMD.parameters(), lr=5e-5) if FMD else None
    list_opt = [opt_ssf, opt_JD, opt_NEW, opt_FMD] 

    for epoch in range(total_epochs):
        a = time.time()
        set_models_state(list_models, 'train', FMD, JD, NEW)
        for i,x in enumerate(iter(train_loader)):
            #Set 0 gradient
            set_opt_zero(list_opt, FMD, JD, NEW)

            #Get the data
            x = x.permute(0, 4, 1, 2, 3)
            x = x.cuda().float()
            x_cur = x[:,:,1,...]
            with torch.no_grad():
                hx = encoder(x[:,:,0,...])[0]
                x_ref = decoder(hx).detach()
                x_1_hat = decoder_hat(hx).detach()
                x_hat = ssf(x_cur, x_ref, x_1_hat)

            # optimize discriminator_JD
            if JD:
                fake_vid = torch.cat((x_1_hat, x_hat), dim = 1)
                real_vid = x[:,0,:2,...].detach() 
                JD_fake_validity = discriminator_JD(fake_vid.detach())
                JD_real_validity = discriminator_JD(real_vid)
                gradient_penalty = compute_gradient_penalty(discriminator_JD, real_vid.data, fake_vid.data)
                errJD =  -torch.mean(JD_real_validity) + torch.mean(JD_fake_validity) + lambda_gp * gradient_penalty
                errJD.backward()
                opt_JD.step()

            # optimize discriminator_NEW
            if NEW:
                fake_vid = torch.cat((x_1_hat, x_hat), dim = 1)
                real_vid_NEW = torch.cat((x_1_hat, x_cur), dim = 1).detach()
                NEW_fake_validity = discriminator_NEW(fake_vid)
                NEW_real_validity = discriminator_NEW(real_vid_NEW)
                gradient_penalty_NEW = compute_gradient_penalty(discriminator_NEW, real_vid_NEW.data, fake_vid.data)
                errNEW =  -torch.mean(NEW_real_validity) + torch.mean(NEW_fake_validity) + lambda_gp * gradient_penalty_NEW
                errNEW.backward()
                opt_NEW.step()

            # optimize discriminator_FMD
            if FMD:
                fake_img = x_hat.detach()
                real_img = x[:,0,:1,...].detach()
                FMD_fake_validity = discriminator_FMD(fake_img)
                FMD_real_validity = discriminator_FMD(real_img)
                gradient_penalty_FMD = compute_gradient_penalty(discriminator_FMD, fake_img.data, real_img.data)
                errFMD =  -torch.mean(FMD_real_validity) + torch.mean(FMD_fake_validity) + lambda_gp * gradient_penalty_FMD
                errFMD.backward()
                opt_FMD.step()
            

            if i%skip_fq == 0:
                x_cur = x_cur.detach()
                x_ref = x_ref.detach()
                x_1_hat = x_1_hat.detach()
                x_hat = ssf(x_cur, x_ref, x_1_hat)
                
                if JD:
                    fake_vid = torch.cat((x_1_hat, x_hat), dim = 1)
                    fake_validity_JD = discriminator_JD(fake_vid)
                    errJD_ = -torch.mean(fake_validity_JD)
                else:
                    errJD_ = 0
                    
                if NEW:
                    fake_vid = torch.cat((x_1_hat, x_hat), dim = 1)
                    fake_validity_NEW = discriminator_NEW(fake_vid)
                    errNEW_ = -torch.mean(fake_validity_NEW)
                else:
                    errNEW_ = 0
                    
                if FMD:
                    fake_img = x_hat
                    fake_validity_im = discriminator_FMD(fake_img)
                    errFMD_ = -torch.mean(fake_validity_im)
                else:
                    errFMD_ = 0

                loss = lambda_MSE*mse(x_hat, x_cur) + lambda_JD*errJD_ + lambda_NEW*errNEW_ + lambda_FMD*errFMD_
                loss.backward()
                opt_ssf.step()

        
        if ((epoch+1) % 10 == 0) or epoch == 0:
            f = open('./saved_models/'+ folder_name + "/performance.txt", "a")
            JD_loss, MSE_loss, NEW_loss, FMD_loss = cal_W1(ssf, encoder, decoder, decoder_hat, discriminator_JD, discriminator_NEW, discriminator_FMD, test_loader, list_models, FMD, JD, NEW)
            show_str = f'   {epoch+1}   {JD_loss.item()} {FMD_loss.item()} {NEW_loss.item()} {MSE_loss.item()}'
            f.write(show_str + "\n")
            f.close()
            print(print_str)
            print(show_str)
            set_models_state(list_models, 'eval', FMD, JD, NEW)
            torch.save(ssf.motion_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_enc.pth'))
            torch.save(ssf.motion_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_dec.pth'))
            torch.save(ssf.P_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'p_enc.pth'))
            torch.save(ssf.res_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_enc.pth'))
            torch.save(ssf.res_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_dec.pth' ))
            if JD:
                torch.save(discriminator_JD.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_JD.pth'))
            if NEW:
                torch.save(discriminator_NEW.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_NEW.pth'))
            if FMD:
                torch.save(discriminator_FMD.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_FMD.pth'))
    
    end = time.time()
    set_models_state(list_models, 'eval', FMD, JD, NEW)
    torch.save(ssf.motion_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_enc.pth'))
    torch.save(ssf.motion_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_dec.pth'))
    torch.save(ssf.P_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'p_enc.pth'))
    torch.save(ssf.res_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_enc.pth'))
    torch.save(ssf.res_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_dec.pth' ))
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
