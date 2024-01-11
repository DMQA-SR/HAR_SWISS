import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, args):
        super(GRUEncoder, self).__init__()
        
        self.args = args
        self.fc = nn.Linear(args.gru_hid_dim*2, args.gru_hid_dim)
        self.attn_fc = nn.Linear(args.gru_hid_dim*2, args.gru_hid_dim*2)

        self.gru = nn.GRU(args.gru_input_size, args.gru_hid_dim, 
                            args.gru_layers, batch_first=True,
                            dropout=args.gru_dropout, bidirectional=args.bidirectional)
        
    def forward(self, x):
        """x : B, L, input_dim"""
        gru_out, _ = self.gru(x) # B, L, input_dim -> B, L, 2*H
        
        h_n = gru_out[:,-1,:] # B, 2*H
        signal_emb = self.fc(h_n) # B, 2*H -> B, H
            
        return signal_emb


class Multi_GRU(nn.Module):
    def __init__(self, args):
        super(Multi_GRU, self).__init__()
        
        self.args = args
        self.gru_emb = GRUEncoder(args)
        
        self.gru_dict = nn.ModuleDict()
        for i in range(args.num_signals):
            self.gru_dict[str(i)] = self.gru_emb
            
    def forward(self, x):
        """x : B, S, L"""
        B, S, _ = x.shape
        
        signals = torch.chunk(x, self.args.num_signals, dim=1) # B, S, L -> B, 1, L
        emb_out = torch.zeros(size=(B, S, self.args.emb_dim), 
                              device=self.args.cuda) # B, S, H
        
        for i, signal in enumerate(signals):
            # B, 1, L -> B, -1, gru input dim
            
            signal = signal.view(B, -1, self.args.gru_input_size).to(self.args.cuda)
            
            # B, -1, gru input dim -> B, H
            signal_emb = self.gru_dict[str(i)](signal)
            emb_out[:,i,:] = signal_emb
            
        return emb_out
                
            
            