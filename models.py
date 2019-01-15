from torch import nn
import torch

class CNN(nn.Module):
    """ A baseline CNN autoencoder for testing"""
    def __init__(self, in_shape):
        super(CNN, self).__init__()
        c,_,_ = in_shape[1:]
        self.encoder = nn.Sequential(nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=4,
                                               stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=4,
                                               stride=2, padding=1),
                                     nn.ReLU()
                                    )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(
                                         64, 32, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(
                                         32, c, kernel_size=4, stride=2, padding=1),
                                     nn.Sigmoid()
                                    )
        if in_shape[2:] == (64, 64):
            self.encoder = nn.Sequential(nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(32, 64, kernel_size=4,
                                               stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=4,
                                               stride=2, padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(
                                        64, 64, kernel_size=4, stride=2, padding=1),
                                     nn.LeakyReLU()
                                        )
            self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(
                                             64, 64, kernel_size=4, stride=2, padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(
                                             64, 32, kernel_size=4, stride=2, padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(
                                             32, c, kernel_size=4, stride=2, padding=1),
                                         nn.Sigmoid()
                                         )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

class VAE(nn.Module):

    def __init__(self, in_shape, nb_latents):
        super(VAE, self).__init__()

        c,_,_ = in_shape[1:]

        self.relu = nn.ReLU()

        encoder_layers = [
                        nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                        nn.ReLU()
        ]
        if in_shape[2:] == (64, 64):
            encoder_layers += [
                        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU()
            ]

        self.encoder_pass = nn.Sequential(*encoder_layers)

        self.encoder_fc = nn.Linear(64*4*4, 256)

        self.fc_mean = nn.Linear(256, nb_latents)
        self.fc_logvar = nn.Linear(256, nb_latents)

        self.decoder_fc1 = nn.Linear(nb_latents, 256)
        self.decoder_fc2 = nn.Linear(256, 64*4*4)

        self.decoder_pass = nn.Sequential(
                        nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),
                        nn.Sigmoid()
        )

        if in_shape[2:] == (64, 64):
            self.decoder_pass = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )

    def encoder(self, x):
        x = self.encoder_pass(x)
        x = self.relu(self.encoder_fc(x.view(-1, 64*4*4)))
        return self.fc_mean(x), self.fc_logvar(x)

    def decoder(self, z):
        out = self.relu(self.decoder_fc1(z))
        out = self.relu(self.decoder_fc2(out))
        out = self.decoder_pass(out.view(-1, 64, 4, 4))
        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, mu, logvar, z

class Discriminator(nn.Module):
    def __init__(self, nb_latents):
        super(Discriminator, self).__init__()
        self.z_dim = nb_latents
        self.D = nn.Sequential(
            nn.Linear(self.z_dim, 1000),
            nn.LeakyReLU(True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(True),
            # nn.Linear(1000, 1000),
            # nn.LeakyReLU(True),
            # nn.Linear(1000, 1000),
            # nn.LeakyReLU(True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(True),
            nn.Linear(1000, 1),
            nn.ReLU()
        )

    def forward(self, z):
        return self.D(z).squeeze()