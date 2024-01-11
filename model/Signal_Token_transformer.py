import torch, os, sys
sys.path.append(os.getcwd())

from torch import nn

from einops import repeat
from model.gru_emb import Multi_GRU
from model.Transformer import *
from utils.init_weights import initialize_weights


class STT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.embedding_layers = Multi_GRU(args)
        self.emb_dropout = nn.Dropout(args.emb_dropout)
        self.transformer_encoder = Transformer(args.emb_dim, args.depth, args.heads, args.head_dim, 
                                               args.transformer_mlp_dim, args.dropout)

        self.sensor_emb = nn.Parameter(torch.randn(1, self.args.num_signals, args.emb_dim), requires_grad=True)
            
        self.apply(initialize_weights)
        
    def forward(self, signals):
        """
        signals shape : (batch, # of signals, data length)"""

        x = self.embedding_layers(signals) # x : (batch, # of signals, emb_dim)
        b, _, _ = x.shape

        if self.args.signal_emb:
            # sensor_tokens : (1, num_signals, dim) -> (batch, num_signals, dim)
            sensor_embedding = repeat(self.sensor_emb, '() n d -> b n d', b = b)
            
            # x : (batch, # of signals, dim) -> (batch, num_signals, dim)
            x += sensor_embedding
        
        x = self.emb_dropout(x)

        # x : (batch, num_signals, dim) -> (batch, num_signals, dim)
        enc_out = self.transformer_encoder(x)
        mean_out = enc_out.mean(dim=1) # batch, dim
        
        features = {'enc_out':enc_out, 'mean_out':mean_out}
        
        return features