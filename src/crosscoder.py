import torch
import torch.nn as nn

class CrossCoder(nn.Module):
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder_with = nn.Linear(dict_size, activation_dim, bias=True)
        self.decoder_without = nn.Linear(dict_size, activation_dim, bias=True)
        self.relu = nn.ReLU()

    def encode(self, x):
        return self.relu(self.encoder(x))

    def forward(self, x_with, x_without):
        f_with = self.encode(x_with)
        f_without = self.encode(x_without)

        recon_with = self.decoder_with(f_with)
        recon_without = self.decoder_without(f_without)

        return recon_with, recon_without, f_with, f_without


def crosscoder_loss(x_with, x_without, recon_with, recon_without, f_with, f_without, l1_coef=1e-3):
    recon_loss = ((x_with - recon_with) ** 2).mean() + ((x_without - recon_without) ** 2).mean()
    sparsity_loss = f_with.abs().mean() + f_without.abs().mean()
    total = recon_loss + l1_coef * sparsity_loss
    return total, recon_loss, sparsity_loss
