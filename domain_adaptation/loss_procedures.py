import torch
import torch.nn.functional as F
from models.LUSR import forward_loss, backward_loss
from models.common import vae_loss, kl_loss

def LUSR_loss(cfg, model, imgs_list, device):
    floss = 0
    img_count = 0

    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        for i in range(0, len(imgs_list)):
            corrupt_imgs = imgs_list[i].type(torch.FloatTensor).to(device)
            floss += forward_loss(corrupt_imgs, model, cfg['beta'])
            img_count += corrupt_imgs.shape[0]
        floss = floss / img_count

        # backward circle
        all_imgs = torch.cat(imgs_list[1:], 0).type(torch.FloatTensor).to(device)
        bloss = backward_loss(all_imgs, model, device)
    else:
        floss += forward_loss(imgs_list, model, cfg['beta'])
        img_count += imgs_list.shape[0]
        floss = floss / img_count
        bloss = backward_loss(imgs_list, model, device)

    return (floss + bloss * cfg['bloss_coeff'])

def VAE_loss(cfg, model, imgs_list, device):
    all_imgs = handle_reshape(imgs_list, device)
    mu, sigma, recon_imgs = model.forward(all_imgs)
    return vae_loss(all_imgs, mu, sigma, recon_imgs, cfg['beta'])

def DVAE_loss(cfg, model, imgs_list, device):
    loss = 0
    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        clean_imgs = imgs_list[0].to(device)

        for i in range(1, len(imgs_list)):
            noised_imgs = imgs_list[i].to(device)
            mu, sigma, recon_imgs = model.forward(noised_imgs)
            loss += vae_loss(clean_imgs, mu, sigma, recon_imgs, cfg['beta'])

        return loss
    else:
        raise ValueError(f"Was only given clean images for DVAE")

def AE_loss(model, imgs_list, device, **kwargs):
    all_imgs = handle_reshape(imgs_list, device)    
    recon = model.forward(all_imgs)
    return torch.nn.functional.mse_loss(recon, all_imgs)

def DARLA_loss(cfg, model, imgs_list, device):
    all_imgs = handle_reshape(imgs_list, device)
    mu, sigma, recon_x = model(all_imgs)
    kl = kl_loss(all_imgs, mu, sigma, cfg['beta'])
    return torch.nn.functional.mse_loss(recon_x, all_imgs) + kl


# def DDVAE_loss(cfg, model, imgs_list, device):
#     loss = 0
#     if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
#         clean_in = imgs_list[0].to(device)
#         c_mu, c_sigma, recon_x, n_mu, n_sigma, noise_recon_x = model.forward(clean_in)

#         clean_only_loss = vae_loss(clean_in, c_mu, c_sigma, recon_x, cfg['beta']) + torch.norm(n_mu, 1) + torch.norm(n_sigma, 1)
#         loss += clean_only_loss

#         for i in range(1, len(imgs_list)):
#             noise_in = imgs_list[i].to(device)
#             pure_noise = noise_in - clean_in
#             c_mu_1, c_sigma_1, recon_x, n_mu, n_sigma, noise_recon_x = model.forward(noise_in)

#             clean_loss = vae_loss(clean_in, c_mu, c_sigma, recon_x, cfg['beta'])
#             noise_loss = vae_loss(pure_noise, n_mu, n_sigma, noise_recon_x, cfg['beta'])

#             param_loss = torch.norm(c_mu - c_mu_1, 1) + torch.norm(c_sigma - c_sigma_1, 1)
#             loss += clean_loss + noise_loss + param_loss
#     else:
#         c_mu, c_sigma, recon_x, n_mu, n_sigma, noise_recon_x = model.forward(imgs_list)

#         clean_only_loss = vae_loss(imgs_list, c_mu, c_sigma, recon_x, cfg['beta']) + torch.norm(n_mu, 1) + torch.norm(n_sigma, 1)
#         loss = clean_only_loss
    
#     return loss

# TODO: Best results so far
# def DDVAE_loss(cfg, model, imgs_list, device):
#     loss = 0
#     if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
#         clean_imgs = imgs_list[0].to(device)
#         # mu_list = list()
#         # sigma_list = list()
#         latent_list = list()

#         for i in range(len(imgs_list)):
#             noised_imgs = imgs_list[i].to(device)
#             mu, sigma, recon_imgs, latent_1 = model.forward(noised_imgs, True)

#             # True VAE
#             loss += vae_loss(clean_imgs, mu, sigma, recon_imgs, cfg['beta'])

#             # DARLA inspired version
#             # loss += torch.nn.functional.mse_loss(recon_imgs, noised_imgs)
#             loss += kl_loss(noised_imgs, mu, sigma, cfg['beta'])

#             # latent_list.append(latent_1)
            
#             mu2, sig2, _, latent_2 = model.forward(recon_imgs, True)
#             latent_list.append(latent_2)

#             # _, _, _, latent_2 = model.forward(recon_imgs, True)
#             # latent_list.append(latent_2)

#         #     mu_list.append(mu)
#         #     sigma_list.append(sigma)
        
#         # param_len = len(mu_list)
#         param_len = len(latent_list)

#         # for i in range(param_len):
#         # mu_1 = mu_list[i]
#         # sigma_1 = sigma_list[i]
#         latent_1 = latent_list[0]
#         for j in range(1, param_len):
#             latent_2 = latent_list[j]
#             loss += F.l1_loss(latent_1, latent_2)
#             # mu_2 = mu_list[j]
#             # sigma_2 = sigma_list[j]
#             # loss += F.l1_loss(mu_1, mu_2) + F.l1_loss(sigma_1, sigma_2)
#     else:
#         raise ValueError(f"Was only given clean images for DVAE")
    
#     return loss


def DDVAE_loss(cfg, model, imgs_list, device):
    loss = 0
    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        clean_imgs = imgs_list[0].to(device)
        mu_list = list()
        sigma_list = list()

        for i in range(len(imgs_list)):
            noised_imgs = imgs_list[i].to(device)
            mu, sigma, recon_imgs, latent_1 = model.forward(noised_imgs, True)

            # True VAE
            loss += vae_loss(clean_imgs, mu, sigma, recon_imgs, cfg['beta'])
            loss += kl_loss(noised_imgs, mu, sigma, cfg['beta'])

            # latent_list.append(latent_1)
            
            # mu2, sig2, _, latent_2 = model.forward(recon_imgs, True)
            
            # mu_list.append(mu2)
            # sigma_list.append(sig2)

            # _, _, _, latent_2 = model.forward(recon_imgs, True)
            # latent_list.append(latent_2)

            mu_list.append(mu)
            sigma_list.append(sigma)
        
        param_len = len(mu_list)

        # for i in range(param_len):
        # mu_1 = mu_list[i]
        # sigma_1 = sigma_list[i]
        mu_c = mu_list[0]
        sigma_c = sigma_list[0]
        for j in range(1, param_len):
            mu_j = mu_list[j]
            sigma_j = sigma_list[j]
            loss += F.l1_loss(mu_c, sigma_j)
            loss += F.l1_loss(sigma_c,sigma_j)
            # mu_2 = mu_list[j]
            # sigma_2 = sigma_list[j]
            # loss += F.l1_loss(mu_1, mu_2) + F.l1_loss(sigma_1, sigma_2)
    else:
        raise ValueError(f"Was only given clean images for DVAE")
    
    return loss



def handle_reshape(imgs_list, device):
    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        return torch.cat(imgs_list, 0).to(device)
    else:
        return imgs_list.to(device)