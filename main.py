import os
import shutil
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from logger import Logger

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

from models import VAE, Discriminator
from data_utils import get_dataloader, get_mnist, get_chairs_dataloader, get_chairs_test_dataloader

def _make_results_dir(dirpath='results'):
    if os.path.isdir('results'):
        shutil.rmtree('results')
    os.makedirs('results')


def traverse(model, datapoint, nb_latents, epoch_nb, batch_idx, dirpath='results'):
    model.eval()
    datapoint = datapoint[0]
    datapoint = datapoint.unsqueeze(0)
   
    mu, _ = model.encoder(datapoint)
    
    recons = torch.zeros((nb_latents, 7, 32, 32))
    for zi in range(nb_latents):
       muc = mu.squeeze().clone()
       
       for i, val in enumerate(np.linspace(-3, 3, 7)):
           muc[zi] = val
           recon = model.decoder(muc).cpu()
           recons[zi, i] = recon.view(32, 32)
  
    filename = os.path.join(dirpath, 'traversal_' +
                            str(epoch_nb) + '_' + str(batch_idx) + '.png')
    save_image(recons.view(-1, 1, 32, 32), filename,
               nrow=nb_latents, pad_value=1)


def permute(z):
    perm_z = []
    for z_j in z.split(1,1):
        perm = torch.randperm(z.size(0))
        if z.is_cuda:
            perm = perm.cuda()
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

def original_vae_loss(reconstructions, x, mu, logvar):
    bce = F.binary_cross_entropy(
        reconstructions, x, size_average=False)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.sum(1).mean(), bce + kld.sum(), bce

def beta_vae_loss(reconstructions, x, mu, logvar, beta):
    """Reconstruction + KL divergence losses summed over all elements and batch."""
    bce = F.binary_cross_entropy(
        reconstructions, x, size_average=False)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.sum(), bce + beta*kld.sum(), bce


def controlled_vae_loss(reconstructions, x, mu, logvar, beta, C):
    """Reconstruction + KL divergence losses summed over all elements and batch."""
    bce = F.binary_cross_entropy(
        reconstructions, x, size_average=False)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    control = kld.sum(1)
    control = torch.abs(control - C)
    return kld.sum(1).mean(), bce + beta*control.sum(), bce


def _tc_loss(z, D):
    """E_q(z)[log D(z) / (1-D(z))]
    Args:
        z: latent code
        D: Discriminator
    """
    D_z = D(z)
    # loss = 0
    # print(torch.log(D_z/(1-D_z)))
    # for i in range(D_z.size(0)):
    #     # print(torch.log(D_z[i] / (1 - D_z[i])))
    #     if torch.log(D_z[i] / (1 - D_z[i])) >= -100 and torch.log(D_z[i] / (1 - D_z[i])) <=100:
    #         loss += torch.log(D_z[i] / (1 - D_z[i]))
    #     else:
    #         loss += 0
    # loss /= D_z.size(0)
    # print(torch.log(D(z)/(1-D(z))))
    # loss = torch.log(D(z)/(1-D(z))).mean()
    loss = D(z).sum()
    return loss

def factor_vae_loss(reconstructions, x, mu, logvar, z, D, gamma):
    bce = F.binary_cross_entropy(
        reconstructions, x, size_average=False)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    tc_loss = _tc_loss(z, D)
    return kld.sum(1).mean(), bce + kld.sum() + gamma * tc_loss, tc_loss, bce 

def controlled_factor_vae_loss(reconstructions, x, mu, logvar, z, D, gamma, C=0.5):
    """C is gradually increasing during training"""
    bce = F.binary_cross_entropy(
        reconstructions, x, size_average=False)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    control = kld.sum(1)
    control = torch.abs(control - C)
    tc_loss = _tc_loss(z, D)
    return kld.sum(1).mean(), bce + control.sum() + gamma * tc_loss, tc_loss, bce

         
def process(model, batch_size, beta, epochs, optimizer, train_loader, test_loader, device):           
    train_loss_ = []
    train_reloss = []
    train_kld = []
    test_loss_ = []
    test_reloss = []
    for epoch in range(epochs):
        
        # Training
        model.train()

        train_loss = 0
        reconstruction_loss = 0
        for batch_i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()

            reconstructions, mu, log_var, z = model(images)
            if batch_i == 0:
                print('r.shape: {} mu.shape:{} var.shape:{} z.shape:{}'.format(
                    reconstructions.shape, mu.shape, log_var.shape, z.shape))

            kld, loss, re_loss = beta_vae_loss(reconstructions, images, mu, log_var, beta)
            reconstruction_loss += re_loss.item()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_i % 10 == 0:
                print('train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  kld: {:.0f}  reLoss: {}'.format(epoch+1, batch_i*len(images),
                                                                                                        len(train_loader.dataset), 100. * batch_i / len(
                                                                                                            train_loader), loss.item() / len(images),
                                                                                                        kld, re_loss))

        print('===> Epoch: {}  Average loss: {:.4f}  Reconstruction loss: {:.4f}'.format(
            epoch+1, train_loss / len(train_loader.dataset), reconstruction_loss / len(train_loader.dataset)))

        # testing
        model.eval()
        test_loss = 0
        test_re_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar, z = model(data)
                _, loss, re_loss = beta_vae_loss(recon_batch, data, mu, logvar, beta)
                test_loss += loss.item()
                test_re_loss += re_loss.item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                    recon_batch[:n]])
                    save_image(comparison.cpu(), 'results/reconstruction_'+str(epoch)+'.png', nrow=n)

        with torch.no_grad():
            sample = torch.randn(64, z.size(1)).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample, 'results/sample_'+str(epoch)+'.png')

        # Using Tensorboard
        # info = {'loss': train_loss / len(train_loader.dataset),
        #         'recon_loss': reconstruction_loss / len(train_loader.dataset),
        #         'kld': kld.item(),
        #         'test loss': test_loss / len(test_loader.dataset),
        #         'test recon_loss': test_re_loss / len(test_loader.dataset)}
        # step = epoch+1
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, step)

        train_loss_.append(train_loss / len(train_loader.dataset))
        train_reloss.append(reconstruction_loss / len(train_loader.dataset))
        train_kld.append(kld.item())
        test_loss_.append(test_loss / len(test_loader.dataset))
        test_reloss.append(test_re_loss / len(test_loader.dataset))
    
    # do latent traversal
    traverse(model, data, 20, epochs, 100)

    plt.subplot(2,2,1)
    plt.plot(train_loss_)
    plt.subplot(2,2,2)
    plt.plot(train_reloss)
   
    plt.subplot(2,2,3)
    plt.plot(test_loss_)
    plt.subplot(2,2,4)
    plt.plot(test_reloss)
    plt.show()


def process_cvae(model, batch_size, beta, epochs, optimizer, train_loader, test_loader, device):
    train_loss_ = []
    train_reloss = []
    train_kld = []
    test_loss_ = []
    test_reloss = []
    for epoch in range(epochs):

        # Training
        model.train()

        train_loss = 0
        reconstruction_loss = 0
        for batch_i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()

            reconstructions, mu, log_var, z = model(images)
            if batch_i == 0:
                print('r.shape: {} mu.shape:{} var.shape:{} z.shape:{}'.format(
                    reconstructions.shape, mu.shape, log_var.shape, z.shape))

            kld, loss, re_loss = controlled_vae_loss(
                reconstructions, images, mu, log_var, beta, epoch/2)
            reconstruction_loss += re_loss.item()
            train_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

            if batch_i % 10 == 0:
                print('train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  kld: {:.0f}  reLoss: {}'.format(epoch+1, batch_i*len(images),
                                                                                           len(train_loader.dataset), 100. * batch_i / len(train_loader), loss.item() / len(images),
                                                                                            kld, re_loss))

        print('===> Epoch: {}  Average loss: {:.4f}  Reconstruction loss: {:.4f}'.format(
            epoch+1, train_loss / len(train_loader.dataset), reconstruction_loss / len(train_loader.dataset)))

        # testing
        model.eval()
        test_loss = 0
        test_re_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar, z = model(data)
                kld, loss, re_loss = controlled_vae_loss(
                    recon_batch, data, mu, logvar, beta, epoch/2)
                test_loss += loss.item()
                test_re_loss += re_loss.item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                            recon_batch[:n]])
                    save_image(
                        comparison.cpu(), 'results/reconstruction_'+str(epoch)+'.png', nrow=n)

        with torch.no_grad():
            sample = torch.randn(64, z.size(1)).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample, 'results/sample_'+str(epoch)+'.png')

        print('===> Epoch: {}  Test loss: {:.4f}  Test Reconstruction loss: {:.4f}'.format(
            epoch+1, test_loss / len(train_loader.dataset), test_re_loss / len(train_loader.dataset)))


        train_loss_.append(train_loss / len(train_loader.dataset))
        train_reloss.append(reconstruction_loss / len(train_loader.dataset))
        train_kld.append(kld.item())
        test_loss_.append(test_loss / len(test_loader.dataset))
        test_reloss.append(test_re_loss / len(test_loader.dataset))

    plt.subplot(2, 2, 1)
    plt.plot(train_loss_)
    plt.subplot(2, 2, 2)
    plt.plot(train_reloss)

    plt.subplot(2, 2, 3)
    plt.plot(test_loss_)
    plt.subplot(2, 2, 4)
    plt.plot(test_reloss)
    plt.show()


def process_fvae(model, D, batch_size, beta, gamma, control, epochs, optimizer, optimizer_D, train_loader, test_loader, device):
    if control is False:
        train_loss_ = []
        train_reloss = []
        train_kld = []
        test_loss_ = []
        test_reloss = []
        for epoch in range(50):

            # Training
            model.train()
            D.train()
            train_loss = 0
            reconstruction_loss = 0
            for batch_i, (images, _) in enumerate(train_loader):
                images = images.to(device)

                optimizer.zero_grad()

                reconstructions, mu, log_var, z = model(images)
                D_z = D(z)
                
                kld, loss, tc_loss, re_loss = factor_vae_loss(
                    reconstructions, images, mu, log_var, z, D, gamma)
                train_loss += loss.item()
                reconstruction_loss += re_loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()

                _,_,_, z_prim = model(images)
                z_perm = permute(z_prim)
                D_z_perm = D(z_perm.detach())
                ones = torch.ones(D_z.size(0)).to(device)
                zeros = torch.zeros(D_z_perm.size(0)).to(device)
                D_loss = F.binary_cross_entropy_with_logits(torch.cat([D_z, D_z_perm]),
                torch.cat([ones, zeros]))

                optimizer_D.zero_grad()
                D_loss.backward()
                optimizer_D.step()

                if batch_i % 10 == 0:
                    print('train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  kld: {:.0f}  tc: {:.0f}'.format(epoch+1, batch_i*len(images),
                                                                                            len(train_loader.dataset), 100. * batch_i / len(train_loader), loss.item() / len(images), 
                                                                                            kld, tc_loss))

            print('===> Epoch: {}  Average loss: {:.4f}  Reconstruction loss: {:.4f}'.format(
                epoch+1, train_loss / len(train_loader.dataset), reconstruction_loss / len(train_loader.dataset)))

            # testing
            model.eval()
            test_loss = 0
            test_re_loss = 0
            with torch.no_grad():
                for i, (data, _) in enumerate(test_loader):
                    data = data.to(device)
                    reconstructions, mu, logvar, z = model(data)
                    kld, loss, tc_loss, re_loss = factor_vae_loss(
                        reconstructions, data, mu, logvar, z, D, gamma)
                    test_re_loss += re_loss.item()
                    test_loss += loss.item()
                    if i == 0:
                        n = min(data.size(0), 8)
                        comparison = torch.cat([data[:n],
                                                reconstructions[:n]])
                        save_image(
                            comparison.cpu(), 'results/reconstruction_'+str(epoch)+'.png', nrow=n)

            with torch.no_grad():
                sample = torch.randn(64, z.size(1)).to(device)
                sample = model.decoder(sample).cpu()
                save_image(sample, 'results/sample_'+str(epoch)+'.png')
            
            print('===> Epoch: {}  Test loss: {:.4f}  Test Reconstruction loss: {:.4f}'.format(
                epoch+1, test_loss / len(train_loader.dataset), test_re_loss / len(train_loader.dataset)))

            train_loss_.append(train_loss / len(train_loader.dataset))
            train_reloss.append(reconstruction_loss / len(train_loader.dataset))
            train_kld.append(kld.item())
            test_loss_.append(test_loss / len(test_loader.dataset))
            test_reloss.append(test_re_loss / len(test_loader.dataset))

        torch.save(model, 'model_fvae.pt')
        plt.subplot(2, 2, 1)
        plt.plot(train_loss_)
        plt.subplot(2, 2, 2)
        plt.plot(train_reloss)

        plt.subplot(2, 2, 3)
        plt.plot(test_loss_)
        plt.subplot(2, 2, 4)
        plt.plot(test_reloss)
        plt.show()
    else:
        train_loss_ = []
        train_reloss = []
        train_kld = []
        test_loss_ = []
        test_reloss = []
        for epoch in range(50):
    
            # Training
            model.train()
            D.train()
            train_loss = 0
            reconstruction_loss = 0
            for batch_i, (images, _) in enumerate(train_loader):
                images = images.to(device)

                optimizer.zero_grad()

                reconstructions, mu, log_var, z = model(images)
                D_z = D(z)
                
                kld, loss, tc_loss, re_loss = controlled_factor_vae_loss(
                    reconstructions, images, mu, log_var, z, D, gamma, epoch/2)
                train_loss += loss.item()
                reconstruction_loss += re_loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()

                _,_,_, z_prim = model(images)
                z_perm = permute(z_prim)
                D_z_perm = D(z_perm.detach())
                ones = torch.ones(D_z.size(0)).to(device)
                zeros = torch.zeros(D_z_perm.size(0)).to(device)
                D_loss = F.binary_cross_entropy_with_logits(torch.cat([D_z, D_z_perm]),
                torch.cat([ones, zeros]))

                optimizer_D.zero_grad()
                D_loss.backward()
                optimizer_D.step()

                if batch_i % 10 == 0:
                    print('train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  kld: {:.0f}  tc: {:.0f}'.format(epoch+1, batch_i*len(images),
                                                                                            len(train_loader.dataset), 100. * batch_i / len(train_loader), loss.item() / len(images), 
                                                                                            kld, tc_loss))

            print('===> Epoch: {}  Average loss: {:.4f}  Reconstruction loss: {:.4f}'.format(
                epoch+1, train_loss / len(train_loader.dataset), reconstruction_loss / len(train_loader.dataset)))

            # testing
            model.eval()
            test_loss = 0
            test_re_loss = 0
            with torch.no_grad():
                for i, (data, _) in enumerate(test_loader):
                    data = data.to(device)
                    reconstructions, mu, logvar, z = model(data)
                    kld, loss, tc_loss, re_loss = controlled_factor_vae_loss(
                        reconstructions, data, mu, logvar, z, D, gamma, epoch/2)
                    test_loss += loss.item()
                    test_re_loss += re_loss.item()
                    if i == 0:
                        n = min(data.size(0), 8)
                        comparison = torch.cat([data[:n],
                                                reconstructions[:n]])
                        save_image(
                            comparison.cpu(), 'results/reconstruction_'+str(epoch)+'.png', nrow=n)

            with torch.no_grad():
                sample = torch.randn(64, z.size(1)).to(device)
                sample = model.decoder(sample).cpu()
                save_image(sample, 'results/sample_'+str(epoch)+'.png')


            print('===> Epoch: {}  Test loss: {:.4f}  Test Reconstruction loss: {:.4f}'.format(
                epoch+1, test_loss / len(train_loader.dataset), test_re_loss / len(train_loader.dataset)))

            train_loss_.append(train_loss / len(train_loader.dataset))
            train_reloss.append(reconstruction_loss /
                                len(train_loader.dataset))
            train_kld.append(kld.item())
            test_loss_.append(test_loss / len(test_loader.dataset))
            test_reloss.append(test_re_loss / len(test_loader.dataset))

        torch.save(model, 'model_cfvae.pt')
        plt.subplot(2, 2, 1)
        plt.plot(train_loss_)
        plt.subplot(2, 2, 2)
        plt.plot(train_reloss)

        plt.subplot(2, 2, 3)
        plt.plot(test_loss_)
        plt.subplot(2, 2, 4)
        plt.plot(test_reloss)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train beta-VAE on the sprites dataset')
    parser.add_argument('--dataset', type=str, default="MNIST")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                        help='learning rate for Adagrad (default: 1e-2)')
    parser.add_argument('--beta', type=int, default=4, metavar='B',
                        help='the beta coefficient (default: 4)')
    parser.add_argument('--gamma', type=int, default=3.5, metavar='B',
                        help='the gamma coefficient (default: 4)')
    parser.add_argument('--nb-latents', type=int, default=20, metavar='N',
                        help='number of latents (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--path', type=str, default=' ')
    parser.add_argument('--model', type=str, default='bvae')
    args = parser.parse_args()

    print('===> Loading Data')
    if args.dataset.upper() == "MNIST":
        if args.path != ' ':
            train_loader, test_loader = get_mnist(args.batch_size, args.path)
        else:
            train_loader, test_loader = get_mnist(args.batch_size)
    else:
        if args.path != ' ':
            train_loader, test_loader = get_chairs_dataloader(args.batch_size, args.path), get_chairs_test_dataloader(args.batch_size, args.path)
        else:
            train_loader, test_loader = get_chairs_dataloader(
                args.batch_size), get_chairs_test_dataloader(args.batch_size)
    logger = Logger('./logs')
    print('===> Data loaded')

    for _, (images,_) in enumerate(train_loader):
        break

    in_shape = images.shape

    model = VAE(in_shape, args.nb_latents).to(args.device)
    D = Discriminator(args.nb_latents).to(args.device)
    # Are we using GPU
    print('\nDeivce: {}'.format(args.device))

    # Print model architecture and parameters
    print('Model architectures:\n{}\n'.format(model))
    print('Parameters and size:')
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, list(param.size())))

    optimizer = optim.Adam(model.parameters(), args.lr)
    optimizer_D = optim.Adam(D.parameters(), args.lr)
    _make_results_dir()
    if args.model.lower() == 'bvae':
        process(model, args.batch_size, args.beta, args.epochs, optimizer, train_loader, test_loader, args.device)
    elif args.model.lower() == 'factorvae':
        process_fvae(model, D, args.batch_size, args.beta, args.gamma, False, args.epochs,
                optimizer, optimizer_D, train_loader, test_loader, args.device)
    elif args.model.lower() == 'cfactorvae':
        process_fvae(model, D, args.batch_size, args.beta, args.gamma, True, args.epochs,
                     optimizer, optimizer_D, train_loader, test_loader, args.device)
    elif args.model.lower() =='cvae':
        process_cvae(model, args.batch_size, args.beta, args.epochs,
                     optimizer, train_loader, test_loader, args.device)

    # save the trained model
    torch.save(model, 'model_%s.pt' % (args.model))
