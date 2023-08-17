import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        self.recon_loss = torch.nn.L1Loss().cuda()

    def __call__(self, output_recons, target_recons):
        recon_loss = self.recon_loss(output_recons, target_recons)

        return recon_loss
    
    