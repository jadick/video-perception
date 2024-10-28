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
parser.add_argument('--lambda_P', type = float, default = 0.0, help = 'Joint perceptual penalty')
parser.add_argument('--lambda_MSE', type = float, default = 1.0, help = 'MSE Penalty')
parser.add_argument('--lambda_PN', type = float, default = 0.0, help = 'New perceptual penalty')
parser.add_argument('--lambda_PM', type = float, default = 0.0, help = 'Marginal perceptual penalty')
parser.add_argument('--path', type = str, default = './data/', help = 'Data path')
parser.add_argument('--pre_path', type = str, default = './fixed_models/', help ='Path to pretrained weights')

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

def cal_W1(ssf, encoder, decoder, decoder_hat, discriminator, discriminator_N, discriminator_M, test_loader, list_models):
    mse_loss = nn.MSELoss(reduction='sum')
    mse_avg = nn.MSELoss()
    set_models_state(list_models, 'eval')

    W1_distance = []
    W1N_distance = []
    W1M_distance = []
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


            fake_vid = torch.cat((x_1_hat, x_hat), dim = 1).detach()
            real_vid = x[:,0,:2,...].detach() #this looks good!
            fake_validity = discriminator(fake_vid)
            real_validity = discriminator(real_vid)

            real_vid_new = torch.cat((x_1_hat, x_cur), dim = 1).detach()
            new_metric_real_validity = discriminator_N(real_vid_new)
            new_metric_fake_validity = discriminator_N(fake_vid)
            
            fake_img = x_hat.detach()
            real_img = x[:,0,6:7,...].detach()
            fake_valid_m = discriminator_M(fake_img)
            real_valid_m = discriminator_M(real_img)

            W1_distance.append(torch.sum(real_validity) - torch.sum(fake_validity))
            W1N_distance.append(torch.sum(new_metric_real_validity) - torch.sum(new_metric_fake_validity))
            W1M_distance.append(torch.sum(real_valid_m) - torch.sum(fake_valid_m))
            MSE.append(mse_loss(x[:,:,1,:,:], x_hat))
            num_x += len(x)

    W1_distance = torch.Tensor(W1_distance)
    MSE = torch.Tensor(MSE)
    W1N_distance = torch.Tensor(W1N_distance)
    W1M_distance = torch.Tensor(W1M_distance)
    finalW1_N =  W1N_distance.sum()/num_x

    finalW1 = W1_distance.sum()/num_x
    finalW1_N =  W1N_distance.sum()/num_x
    finalW1_M =  W1M_distance.sum()/num_x
    finalMSE = MSE.sum()/(64*64*num_x)
    

    return finalW1, finalMSE, finalW1_N, finalW1_M

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
    lambda_P = args.lambda_P * 1e-3
    lambda_PN = args.lambda_PN * 1e-3
    lambda_PM = args.lambda_PM * 1e-3
    lambda_MSE = args.lambda_MSE
    L = args.L
    path = args.path
    pre_path = args.pre_path
    bit_rate = (z_dim * 2 * np.log2(L)).astype(int)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #No quantization:
    stochastic = True
    quantize_latents = True
    if L == -1:
        stochastic = False
        quantize_latents = False
    
    #Create folder:
    folder_name = f'{bit_rate}/R1-eps|dim_{dim}|z_dim_{z_dim}|L_{L}|lambda_gp_{lambda_gp}|bs_{bs}|dpenalty_{d_penalty}|lambdaP_{lambda_P}|lambdaPM_{lambda_PM}|lambdaPN_{lambda_PN}|lambdaMSE_{lambda_MSE}'
    print ("Settings: ", folder_name)

    os.makedirs('./saved_models/'+ folder_name, exist_ok=True)
    f = open('./saved_models/'+ folder_name + "/performance.txt", "a")
    f.write(f'l_PN, l_P, l_MSE, d_penalty  {lambda_PN} {lambda_P} {lambda_MSE} {d_penalty}\n')
    f.write('| EPOCH | PLF-JD | PLF-FMD | PLF-NEW | MSE |\n')

    #Define Models
    discriminator = Discriminator_v3(out_ch=2).to(device) 
    discriminator_N = Discriminator_v3(out_ch=2).to(device)
    discriminator_M = Discriminator_v3(out_ch=1).to(device)
    ssf = ScaleSpaceFlow_R1eps(num_levels=1, dim=z_dim, stochastic=stochastic, quantize_latents=quantize_latents, L=L).to(device)
    list_models = [discriminator, discriminator_N, discriminator_M, ssf] 

    #Load models:
    if pre_path != 'None':
        print(f'Initializing weights from: {pre_path}')
        ssf.motion_encoder.load_state_dict(torch.load(pre_path + '/m_enc.pth'))
        ssf.motion_decoder.load_state_dict(torch.load(pre_path + '/m_dec.pth'))
        ssf.P_encoder.load_state_dict(torch.load(pre_path + '/p_enc.pth'))
        ssf.res_encoder.load_state_dict(torch.load(pre_path + '/r_enc.pth'))
        ssf.res_decoder.load_state_dict(torch.load(pre_path + '/r_dec.pth'))
        discriminator.load_state_dict(torch.load(pre_path + '/discriminator.pth'))
        discriminator_N.load_state_dict(torch.load(pre_path + '/discriminator_N.pth'))
        discriminator_M.load_state_dict(torch.load(pre_path + '/discriminator_M.pth'))
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
    train_loader, test_loader = get_dataloader(data_root=path, seq_len=8, batch_size=bs, num_digits=1)
    mse = torch.nn.MSELoss()

    #Define optimizers
    opt_ssf= torch.optim.RMSprop(ssf.parameters(), lr=1e-5)
    opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=5e-5)
    opt_dn = torch.optim.RMSprop(discriminator_N.parameters(), lr=5e-5)
    opt_dm = torch.optim.RMSprop(discriminator_M.parameters(), lr=5e-5)
    list_opt = [opt_ssf, opt_d, opt_dn, opt_dm] 

    for epoch in range(total_epochs):
        a = time.time()
        set_models_state(list_models, 'train')
        for i,x in enumerate(iter(train_loader)):
            #Set 0 gradient
            set_opt_zero(list_opt)

            #Get the data
            x = x.permute(0, 4, 1, 2, 3)
            x = x.cuda().float()
            x_cur = x[:,:,1,...]
            with torch.no_grad():
                hx = encoder(x[:,:,0,...])[0]
                x_ref = decoder(hx).detach()
                x_1_hat = decoder_hat(hx).detach()
                x_hat = ssf(x_cur, x_ref, x_1_hat)


            # optimize discriminator
            fake_vid = torch.cat((x_1_hat, x_hat), dim = 1)
            real_vid = x[:,0,:2,...].detach() 
            fake_validity = discriminator(fake_vid.detach())
            real_validity = discriminator(real_vid)
            gradient_penalty = compute_gradient_penalty(discriminator, real_vid.data, fake_vid.data)
            errVD =  -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            errVD.backward()
            opt_d.step()

            # optimize discriminator_N
            real_vid_new = torch.cat((x_1_hat, x_cur), dim = 1).detach()
            new_metric_fake_validity = discriminator_N(fake_vid)
            new_metric_real_validity = discriminator_N(real_vid_new)
            gradient_penalty_N = compute_gradient_penalty(discriminator_N, real_vid_new.data, fake_vid.data)
            errVDN =  -torch.mean(new_metric_real_validity) + torch.mean(new_metric_fake_validity) + lambda_gp * gradient_penalty_N
            errVDN.backward()
            opt_dn.step()

            # optimize discriminator_M
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
                x_1_hat = x_1_hat.detach()
                x_hat = ssf(x_cur, x_ref, x_1_hat)

                fake_vid = torch.cat((x_1_hat, x_hat), dim = 1)
                fake_validity = discriminator(fake_vid)
                errVG = -torch.mean(fake_validity)

                new_metric_fake_validity = discriminator_N(fake_vid)
                errVGN = -torch.mean(new_metric_fake_validity)
                
                fake_img = x_hat
                fake_validity_im = discriminator_M(fake_img)
                errIG = -torch.mean(fake_validity_im)

                loss = lambda_MSE*mse(x_hat, x_cur) + lambda_P*errVG + lambda_PN*errVGN + lambda_PM*errIG
                loss.backward()

                opt_ssf.step()

        
        if ((epoch+1) % 10 == 0) or epoch == 0:
            P_loss, MSE_loss, PN_loss, PM_loss = cal_W1(ssf, encoder, decoder, decoder_hat, discriminator, discriminator_N, discriminator_M, test_loader, list_models)
            show_str = f'{epoch+1} {P_loss.item()} {PM_loss.item()} {PN_loss.item()} {MSE_loss.item()}'
            f.write(show_str + "\n")
            print(show_str)
            set_models_state(list_models, 'eval')
            torch.save(ssf.motion_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_enc.pth'))
            torch.save(ssf.motion_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_dec.pth'))
            torch.save(ssf.P_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'p_enc.pth'))
            torch.save(ssf.res_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_enc.pth'))
            torch.save(ssf.res_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_dec.pth' ))
            torch.save(discriminator.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator.pth'))
            torch.save(discriminator_N.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_N.pth'))
            torch.save(discriminator_M.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_M.pth'))
    
    end = time.time()
    set_models_state(list_models, 'eval')
    torch.save(ssf.motion_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_enc.pth'))
    torch.save(ssf.motion_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'm_dec.pth'))
    torch.save(ssf.P_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'p_enc.pth'))
    torch.save(ssf.res_encoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_enc.pth'))
    torch.save(ssf.res_decoder.state_dict(), os.path.join("./saved_models/" + folder_name, 'r_dec.pth' ))
    torch.save(discriminator.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator.pth'))
    torch.save(discriminator_N.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_N.pth'))
    torch.save(discriminator_M.state_dict(), os.path.join("./saved_models/" + folder_name, 'discriminator_M.pth'))
    f.write(f'Total Training Time: {(end-start)/3600} hours \n')
    f.close()

if __name__ == "__main__":
    main()
