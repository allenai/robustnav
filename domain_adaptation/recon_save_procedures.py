import torch

def LUSR_image_save(model, all_imgs):
    rand_idx = torch.randperm(all_imgs.shape[0])
    imgs1 = all_imgs[rand_idx[:9]]
    imgs2 = all_imgs[rand_idx[-9:]]
    mu, _, classcode1 = model.encoder(imgs1)
    _, _, classcode2 = model.encoder(imgs2)
    recon_imgs1 = model.decoder(torch.cat([mu, classcode1], dim=1))
    recon_combined = model.decoder(torch.cat([mu, classcode2], dim=1))

    return torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)

def VAE_image_save(model, all_imgs):
    rand_idx = torch.randperm(all_imgs.shape[0])
    imgs = all_imgs[rand_idx[:9]]
    _, _, recon = model(imgs)
    return torch.cat([imgs, recon], dim=0)

def AE_image_save(model, all_imgs):
    rand_idx = torch.randperm(all_imgs.shape[0])
    imgs = all_imgs[rand_idx[:9]]
    recon = model(imgs)
    return torch.cat([imgs, recon], dim=0)

def DDVAE_image_save(model, imgs_list, device):
    # inputs_list = None
    # outputs_list = None
    recon_list = None
    if isinstance(imgs_list, list) or isinstance(imgs_list, tuple):
        clean = imgs_list[0]
        rand_idx = torch.randperm(clean.shape[0])
        clean = clean[rand_idx[:9]].to(device)

        # _, _, clean_recon, _, _, clean_noise_recon = model(clean)
        _, _, clean_recon = model(clean)
        recon_list = [clean, clean_recon]
        # inputs_list = [clean]
        # outputs_list = [clean_recon]

        for i in range(1, len(imgs_list)):
            noise = imgs_list[i][rand_idx[:9]].to(device)
            # _, _, noised_clean_recon, _, _, noised_noise_recon = model(noise)
            _, _, noised_clean_recon = model(noise)

            recon_list.append(noise)
            recon_list.append(noised_clean_recon)
            # inputs_list.append(noise)
            # outputs_list.append(noised_clean_recon)        

        # in_tensor = torch.cat(inputs_list, dim=0)
        # out_tensor = torch.cat(outputs_list, dim=0)

        # return torch.cat((in_tensor, out_tensor), dim=0)
        return torch.cat(recon_list, dim=0)
    else:
        rand_idx = torch.randperm(imgs_list)
        clean = clean[rand_idx[:9]].to(device)

        _, _, clean_recon, _, _, clean_noise_recon = model(clean)

        return torch.cat((clean_recon, clean_noise_recon), dim=0)


        