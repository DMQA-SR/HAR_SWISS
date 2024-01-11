from torch import nn
from utils.init_weights import initialize_weights


class MLPHead(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(MLPHead, self).__init__()

        self.hid_dim = args.mlp_hidden
        self.hid_dim2 = int(args.mlp_hidden/2)
        
        self.norm1 = nn.BatchNorm1d(self.hid_dim)
        self.norm2 = nn.BatchNorm1d(self.hid_dim2)
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, self.hid_dim),
            self.norm1,
            nn.ReLU(inplace=True),
            nn.Dropout(args.mlp_dropout)
            )
        
        self.layer2 = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim2),
            self.norm2,
            nn.ReLU(inplace=True),
            nn.Dropout(args.mlp_dropout)
            )

        self.layer3 = nn.Sequential(
            nn.Linear(self.hid_dim2, output_dim)
            )
        self.apply(initialize_weights)

    def forward(self, x):
        """x : batch, emb_dim"""
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        
        """return : batch, feature dim"""
        return out
    
    
class AdditionalTokenHead(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(AdditionalTokenHead, self).__init__()
        
        self.mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                 nn.Linear(input_dim, n_classes))

        self.apply(initialize_weights)
    def forward(self, x):
        return self.mlp(x)
    
    
class Projector(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        flatten_dim = args.emb_dim*args.num_signals

        self.norm = nn.BatchNorm1d(args.proj_hiddim)

        self.net = nn.Sequential(
            nn.Linear(flatten_dim, args.proj_hiddim),
            self.norm, nn.ReLU(inplace=True),
            nn.Linear(args.proj_hiddim, args.proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.norm = nn.BatchNorm1d(args.proj_hiddim)    
                
        self.net = nn.Sequential(
            nn.Linear(args.proj_dim, args.proj_hiddim),
            self.norm, nn.ReLU(inplace=True),
            nn.Linear(args.proj_hiddim, args.proj_dim)
        )
        self.apply(initialize_weights)
        
    def forward(self, x):
        return self.net(x)
    

class ReconstructionHead(nn.Module):
    def __init__(self, num_signals, input_dim, feature_dim):
        super(ReconstructionHead, self).__init__()
        
        self.num_signals = num_signals
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        self.recon = nn.Linear(self.num_signals*self.input_dim,
                            self.num_signals*self.feature_dim)
        self.apply(initialize_weights)
        
    def forward(self, x):
        """x shape : batch, signals, emb dim"""
        x = x.view(-1, self.num_signals*self.input_dim) # batch, num_signals*emb_dim
        x = self.recon(x) # batch, num_signals*feature_dim
        x = x.view(-1, self.num_signals, self.feature_dim) # batch, num_signals, feature dim
        
        return x