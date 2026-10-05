"""Optional PyTorch model used by the A6 CPU experiment."""
import math
import torch
from torch import nn
from abstractgym.a6 import VOCAB


class TinyTransformer(nn.Module):
    def __init__(self, outputs=6, width=64, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), width, padding_idx=0)
        layer = nn.TransformerEncoderLayer(width, 4, dim_feedforward=2*width,
                    dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, outputs)
        pos = torch.arange(256).float().unsqueeze(1)
        freq = torch.exp(torch.arange(0,width,2).float()*(-math.log(10000.)/width))
        pe = torch.zeros(256,width); pe[:,0::2] = torch.sin(pos*freq); pe[:,1::2] = torch.cos(pos*freq)
        self.register_buffer('positions',pe)

    def forward(self, ids):
        x = self.embedding(ids) + self.positions[:ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=ids.eq(0))
        return self.head(self.norm(x[:,0]))


def padded(sequences):
    return nn.utils.rnn.pad_sequence([torch.tensor(x,dtype=torch.long) for x in sequences],batch_first=True,padding_value=0)
