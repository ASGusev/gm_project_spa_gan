from itertools import chain

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import wandb
from tqdm.auto import trange


DESC_TEMPLATE = 'Critic loss: adv {:.3f} gp {:.3f} Gen loss: adv {:.3f} rec {:.3f} fm {:.3f}'


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, int_dim: int, out_dim: int):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_dim, int_dim, 3, padding=1),
            nn.InstanceNorm2d(int_dim),
            nn.PReLU(),
            nn.Conv2d(int_dim, out_dim, 3, padding=1),
            nn.InstanceNorm2d(out_dim)
        )
        self.shortcut = nn.Identity() if in_dim == out_dim \
            else nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1),
                nn.InstanceNorm2d(out_dim)
            )
        self.final_activation = nn.PReLU()

    def forward(self, x):
        return self.final_activation(self.main_path(x) + self.shortcut(x))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(3, 16, 16),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128, 128),
            nn.MaxPool2d(2),
            # ResidualBlock(256, 512, 512),
            # nn.MaxPool2d(2)
        )
        self.bottleneck = ResidualBlock(128, 128, 128)
        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(128, 128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(64, 32, 16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResidualBlock(16, 16, 4),
            nn.Conv2d(4, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def get_fms(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder[0](x)
        x = self.decoder[1](x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


def make_down_block(in_dim: int, int_dim: int, out_dim: int):
    return nn.ModuleList([
       ResidualBlock(in_dim, int_dim, out_dim),
       nn.MaxPool2d(2, 2)
    ])


def make_up_block(in_dim: int, int_dim: int, out_dim: int):
    return nn.ModuleList([
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        ResidualBlock(in_dim, int_dim, out_dim)
    ])


class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_path = nn.ModuleList([
            make_down_block(3, 32, 64),
            make_down_block(64, 128, 128),
            make_down_block(128, 256, 256),
            make_down_block(256, 512, 512)
        ])
        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 512, 512),
            ResidualBlock(512, 512, 512)
        )
        self.up_path = nn.ModuleList([
            make_up_block(1024, 512, 256),
            make_up_block(512, 256, 128),
            make_up_block(256, 128, 64),
            make_up_block(128, 64, 32)
        ])
        self.final_layer = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        states = []
        for block, max_pooling in self.down_path:
            x = block(x)
            states.append(x)
            x = max_pooling(x)
        x = self.bottleneck(x)
        for (upsampling, block), state in zip(self.up_path, states[::-1]):
            x = upsampling(x)
            x = torch.cat((x, state), dim=1)
            x = block(x)
        x = self.final_layer(x)
        return x

    def get_fms(self, x):
        state = None
        for block, max_pooling in self.down_path:
            x = block(x)
            state = x
            x = max_pooling(x)
        x = self.bottleneck(x)
        x = self.up_path[0][0](x)
        x = torch.cat((x, state), dim=1)
        x = self.up_path[0][1](x)
        return x


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_layers = nn.Sequential(
            ResidualBlock(3, 16, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 128, 256),
            nn.MaxPool2d(2),
            ResidualBlock(256, 256, 512),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 1),
            nn.Sigmoid(),
        )
        self.final_clf = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_layers(x).mean(dim=(2, 3))
        return self.final_clf(features)

    def apply_attention(self, x):
        attention = self.feature_layers(x).sum(dim=1)
        attention /= attention.max(dim=2)[0].max(dim=1)[0].unsqueeze(1).unsqueeze(2)
        attention = attention.unsqueeze(1)
        attention = F.interpolate(attention, scale_factor=16)
        return x * attention


def calc_gradient_penalty(critic, real_samples, fake_samples):
    fake_weights = torch.rand((len(real_samples), 1, 1, 1), device=real_samples.device)
    mixed_samples = (1 - fake_weights) * real_samples + fake_weights * fake_samples
    mixed_samples.requires_grad_(True)
    prediction = critic(mixed_samples)
    gradients = torch.autograd.grad(
        outputs=prediction,
        inputs=mixed_samples,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True
    )
    gradients = torch.cat(gradients).view(len(gradients), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def make_wandb_samples(dl, conv_func, device):
    images_original = next(iter(dl))[0]
    images_converted = conv_func(images_original.to(device))
    images_combined = torch.cat((images_original.cpu(), images_converted.cpu()), dim=3)
    images_combined = images_combined.permute(0, 2, 3, 1).numpy()
    return list(map(wandb.Image, images_combined))


class SPAGAN:
    def __init__(self, lambda_fm=1., lambda_rec=10., lambda_gp=10., direct_rec_weight=.3, gen_type=UNetGenerator):
        self.lambda_fm = lambda_fm
        self.lambda_rec = lambda_rec
        self.lambda_gp = lambda_gp
        self.direct_rec_weight = direct_rec_weight
        self.device = 'cpu'
        self.g_a = gen_type()
        self.g_b = gen_type()
        self.d_a = Critic()
        self.d_b = Critic()

    @staticmethod
    def calc_critic_loss(critic, images_real, images_fake):
        critic_input = torch.cat((images_fake, images_real))
        target = torch.cat((torch.ones(len(images_fake), 1), torch.zeros(len(images_real), 1))).to(images_real.device)
        critic_output = critic(critic_input)
        adv_loss = F.binary_cross_entropy(critic_output, target)
        gp_loss = calc_gradient_penalty(critic, images_real, images_fake)
        return adv_loss, gp_loss

    @staticmethod
    def calc_fm_loss(images_src, images_conv, am_desc, fm_gen):
        src_att = images_src * am_desc.apply_attention(images_src)
        conv_att = images_conv * am_desc.apply_attention(images_conv)
        src_fm = fm_gen.get_fms(src_att)
        conv_fm = fm_gen.get_fms(conv_att)
        return torch.abs(src_fm - conv_fm).mean()

    def a2b(self, x):
        return self.g_b(self.d_a.apply_attention(x))

    def b2a(self, x):
        return self.g_a(self.d_b.apply_attention(x))

    def to(self, device):
        self.device = device
        self.g_a.to(device)
        self.g_b.to(device)
        self.d_a.to(device)
        self.d_b.to(device)
        return self

    def save(self, path):
        path.mkdir()
        torch.save(self.g_a.state_dict(), path / 'g_a.pt')
        torch.save(self.g_b.state_dict(), path / 'g_b.pt')
        torch.save(self.d_a.state_dict(), path / 'd_a.pt')
        torch.save(self.d_b.state_dict(), path / 'd_b.pt')

    def load(self, path):
        self.g_a.load_state_dict(torch.load(path / 'g_a.pt', map_location='cpu'))
        self.g_b.load_state_dict(torch.load(path / 'g_b.pt', map_location='cpu'))
        self.d_a.load_state_dict(torch.load(path / 'd_a.pt', map_location='cpu'))
        self.d_b.load_state_dict(torch.load(path / 'd_b.pt', map_location='cpu'))
        self.g_a.to(self.device)
        self.g_b.to(self.device)
        self.d_a.to(self.device)
        self.d_b.to(self.device)

    def calc_gen_loss(self, images_a, images_b):
        conv_a = self.a2b(images_a)
        rest_a = self.b2a(conv_a)
        critic_out_a = self.d_b(conv_a)
        critic_tgt_a = torch.zeros_like(critic_out_a)
        conv_b = self.b2a(images_b)
        rest_b = self.a2b(conv_b)
        critic_out_b = self.d_a(conv_b)
        critic_tgt_b = torch.zeros_like(critic_out_b)

        loss_g_a_adv = F.binary_cross_entropy(critic_out_a, critic_tgt_a)
        loss_g_b_adv = F.binary_cross_entropy(critic_out_b, critic_tgt_b)
        loss_gen_rec = torch.abs(images_a - rest_a).mean() + torch.abs(images_b - rest_b).mean() + \
            self.direct_rec_weight * (torch.abs(self.a2b(images_b) - images_b).mean() + torch.abs(self.b2a(images_a) - images_a).mean())
        loss_fm = SPAGAN.calc_fm_loss(images_a, conv_b, self.d_a, self.g_a) + \
            SPAGAN.calc_fm_loss(images_b, conv_a, self.d_b, self.g_b)

        return loss_g_a_adv, loss_g_b_adv, loss_gen_rec, loss_fm

    def evaluate(self, dl_a, dl_b):
        desc_adv_losses, desc_gp_losses, gen_adv_losses, gen_rec_losses, fm_losses = [], [], [], [], []
        for (images_a,), (images_b,) in zip(dl_a, dl_b):
            images_a, images_b = images_a.to(self.device), images_b.to(self.device)

            with torch.no_grad():
                conv_a, conv_b = self.a2b(images_a), self.b2a(images_b)
            loss_c_b_adv, loss_gp_b = self.calc_critic_loss(self.d_b, images_b, conv_a)
            loss_c_a_adv, loss_gp_a = self.calc_critic_loss(self.d_a, images_a, conv_b)

            with torch.no_grad():
                loss_g_a_adv, loss_g_b_adv, loss_gen_rec, loss_fm = self.calc_gen_loss(images_a, images_b)

            desc_adv_losses.append(loss_c_b_adv.item() + loss_c_a_adv.item())
            desc_gp_losses.append(loss_gp_b.item() + loss_gp_a.item())
            gen_adv_losses.append(loss_g_a_adv.item() + loss_g_b_adv.item())
            gen_rec_losses.append(loss_gen_rec.item())
            fm_losses.append(loss_fm.item())
        desc_adv_loss = np.mean(desc_adv_losses)
        desc_gp_loss = np.mean(desc_gp_losses)
        gen_adv_loss = np.mean(gen_adv_losses)
        rec_loss = np.mean(gen_rec_losses)
        fm_loss = np.mean(fm_losses)
        return desc_adv_loss, desc_gp_loss, gen_adv_loss, rec_loss, fm_loss

    def set_train(self, val):
        self.d_a.train(val)
        self.d_b.train(val)
        self.g_a.train(val)
        self.g_b.train(val)

    def train(self, dl_train_a, dl_train_b, dl_val_a, dl_val_b, n_epochs, lr_critic, lr_gen):
        opt_g = torch.optim.Adam(chain(self.g_a.parameters(), self.g_b.parameters()), lr_gen)
        opt_d = torch.optim.Adam(chain(self.d_a.parameters(), self.d_b.parameters()), lr_critic)
        progress_bar = trange(n_epochs)
        for _ in progress_bar:
            desc_adv_losses, desc_gp_losses, gen_adv_losses, gen_rec_losses, fm_losses = [], [], [], [], []
            self.set_train(True)
            for (images_a,), (images_b,) in zip(dl_train_a, dl_train_b):
                images_a, images_b = images_a.to(self.device), images_b.to(self.device)

                with torch.no_grad():
                    conv_a, conv_b = self.a2b(images_a), self.b2a(images_b)
                opt_d.zero_grad()
                loss_c_b_adv, loss_gp_b = self.calc_critic_loss(self.d_b, images_b, conv_a)
                loss_c_a_adv, loss_gp_a = self.calc_critic_loss(self.d_a, images_a, conv_b)
                loss_desc = (loss_c_b_adv + loss_gp_b + loss_c_a_adv + loss_gp_a)
                loss_desc.backward()
                opt_d.step()

                opt_g.zero_grad()
                loss_g_a_adv, loss_g_b_adv, loss_gen_rec, loss_fm = self.calc_gen_loss(images_a, images_b)
                loss_gen = loss_g_a_adv + loss_g_b_adv + loss_gen_rec * self.lambda_rec + loss_fm * self.lambda_fm
                loss_gen.backward()
                opt_g.step()

                desc_adv_losses.append(loss_c_b_adv.item() + loss_c_a_adv.item())
                desc_gp_losses.append(loss_gp_b.item() + loss_gp_a.item())
                gen_adv_losses.append(loss_g_a_adv.item() + loss_g_b_adv.item())
                gen_rec_losses.append(loss_gen_rec.item())
                fm_losses.append(loss_fm.item())
            train_desc_adv_loss = np.mean(desc_adv_losses)
            train_desc_gp_loss = np.mean(desc_gp_losses)
            train_gen_adv_loss = np.mean(gen_adv_losses)
            train_rec_loss = np.mean(gen_rec_losses)
            train_fm_loss = np.mean(fm_losses)
            self.set_train(False)

            val_critic_adv_loss, val_gp_loss, val_gen_adv_loss, val_rec_loss, val_fm_loss = \
                self.evaluate(dl_val_a, dl_val_b)

            desc = DESC_TEMPLATE.format(val_critic_adv_loss, val_gp_loss, val_gen_adv_loss, val_rec_loss, val_fm_loss)
            progress_bar.set_description(desc)

            with torch.no_grad():
                a2b_samples = make_wandb_samples(dl_val_a, self.a2b, self.device)
                b2a_samples = make_wandb_samples(dl_val_b, self.b2a, self.device)

            wandb.log({
                'Train critic adversarial loss': train_desc_adv_loss,
                'Train critic gradient penalty': train_desc_gp_loss,
                'Train gen adversarial loss': train_gen_adv_loss,
                'Train reconstruction loss': train_rec_loss,
                'Train feature map loss': train_fm_loss,
                'Val critic adversarial loss': val_critic_adv_loss,
                'Val critic gradient penalty': val_gp_loss,
                'Val gen adversarial loss': val_gen_adv_loss,
                'Val reconstruction loss': val_rec_loss,
                'Val feature map loss': val_fm_loss,
                'A2B samples': a2b_samples,
                'B2A samples': b2a_samples
            })
