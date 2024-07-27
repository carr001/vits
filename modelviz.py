import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from text.symbols import symbols
import utils

from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
rank = 0
hps = utils.get_hparams()

device = 'cuda'

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda(rank)

x = torch.randint(0,10,(64,32)).to(device)
xlen = torch.tensor([32]).to(device)
y = torch.randint(0,10,(64, 513, 1)).float().to(device)
ylen = torch.tensor([513]).float().to(device)

writer = SummaryWriter('runs/fashion_mnist_experiment_1')

writer.add_graph(net_g, [x,xlen,y,ylen])
writer.close()


# net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
# images = torch.randint(0,10,(1,32,32)).float()
#writer2 = SummaryWriter('runs/fashion_mnist_experiment_2')

#writer2.add_graph(net_g, images)
#writer2.close()
