import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def initialize_weights(module):
    """Initialize weights for the network layers"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class DAttention(nn.Module):
    """Deep Attention Multiple Instance Learning (D-Attention) module
    
    This module implements a deep attention-based MIL architecture for
    whole slide image classification.
    """
    
    def __init__(self, input_dim=512, n_classes=2, dropout=0.25, act='relu', 
                 rrt=None, n_ctx=4, n_flp=0, num_patch_prompt=0):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        
        # Feature embedding layers
        feature_layers = [nn.Linear(input_dim, 512)]

        if act.lower() == 'gelu':
            feature_layers.append(nn.GELU())
        else:
            feature_layers.append(nn.ReLU())

        if dropout:
            feature_layers.append(nn.Dropout(0.25))

        self.feature = nn.Sequential(*feature_layers)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )
        
        self.apply(initialize_weights)

    def reparameterise(self, mu, logvar):
        """Reparameterization trick for variational autoencoder"""
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def kl_loss(self, mu, logvar):
        """KL divergence loss for variational autoencoder"""
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return torch.mean(kl_div)

    def forward(self, x1, x2=None, label=None, return_attn=False, 
                no_norm=False, pretrain=False, club=None, results_dict=None):
        """Forward pass through the model
        
        Args:
            x1: Input features
            x2: Additional features (optional)
            label: Class labels (optional)
            return_attn: Whether to return attention weights
            no_norm: Whether to skip normalization
            pretrain: Whether in pretraining mode
            club: Club instance (optional)
            results_dict: Dictionary to store intermediate results
            
        Returns:
            Logits, predicted class, probability distribution, and results dictionary
        """
        feature = self.feature(x1)  
        if results_dict is None:
            results_dict = {}

        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone().detach()
        results_dict['patch_attn'] = A_ori
        
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL

        results_dict['slide_feat'] = M

        x = self.classifier(M)

        Y_hat = torch.argmax(x)
        Y_prob = F.softmax(x)
        
        return x, Y_hat, Y_prob, results_dict