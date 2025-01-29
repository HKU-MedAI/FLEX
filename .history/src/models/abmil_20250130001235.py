import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
# from models.prompts import get_prompts
from models.infonce import InfoNCE
from models.vmamba import VSSM
from models.model_rrt import RRTEncoder
from models.mi_estimator import CLUB
from models.tc_estimator import TCLineEstimator
from models.pareto import pareto_fn
from models.conch_custom import TextEncoder, PromptLearner

# conch_model, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path="./models/weights/conch.bin")

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        # loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean
        loss = self.gamma * neg_pairs_mean

        return loss

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # ref from meituan
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        self.model = list(models.resnet50(pretrained = True).children())[:-1]
        self.features = nn.Sequential(*self.model)

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.classifier = nn.Linear(512,1)
        initialize_weights(self.feature_extractor_part2)
        initialize_weights(self.classifier)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x=self.feature_extractor_part2(x)
        # feat = torch.mean(x,dim=0)
        x1 = self.classifier(x)
        # x2 = torch.mean(x1, dim=0).view(1,-1)
        x2,_ = torch.max(x1, dim=0)
        x2=x2.view(1,-1)
        return x2,x
class AttentionGated(nn.Module):
    def __init__(self,input_dim,act='relu',bias=False,dropout=False,rrt=None):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(1024, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        if rrt is not None:
            self.feature += [rrt] 
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        Y_prob = self.classifier(x)

        return Y_prob

class DAttention(nn.Module):
    def __init__(self, input_dim=512, n_classes=2, dropout=0.25, act='relu', rrt=None, n_ctx=4, n_flp=0, num_patch_prompt=0):
        super(DAttention, self).__init__()
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1
        self.feature = [nn.Linear(input_dim, 512)]

        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(0.25)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )
        
        self.apply(initialize_weights)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def forward(self, x1, x2=None, label=None, return_attn=False,no_norm=False, pretrain=False, club=None, results_dict=None):

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