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
from models.abmil import DAttention
import os
from PIL import Image
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 1024*2**20

conch_model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path="./models/weights/conch.bin")
conch_model = conch_model.to("cuda")


class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

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
    
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_inner: int = 0,
                 pre_norm: bool = False, device: torch.device = None,
                 **kwargs):
        super().__init__()

        self.pre_norm = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
        ) if pre_norm else nn.Identity()

        self._real_output_dim = output_size

        self.fc1 = nn.Linear(input_size, hidden_size, device=device)

        blocks = []
        for _ in range(num_inner):
            blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_size, device=device),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size, device=device),
            ))
        self.blocks = nn.ModuleList(blocks)

        self.final = nn.Sequential(
            nn.LayerNorm(hidden_size, device=device),
            nn.GELU(),
            nn.Linear(hidden_size, output_size, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        x = self.fc1(x)
        for block in self.blocks:
            x = x + block(x)
        x = self.final(x)

        return x

class CATEMIL(nn.Module):
    def __init__(self, input_dim=512, n_classes=2, dropout=0.25, act='relu', rrt=None, n_ctx=None, n_flp=0, num_patch_prompt=0, task=None, fold=None, exp_code=None, train=True, base_mil=None, slide_align=1):
        super(CATEMIL, self).__init__()
        self.slide_align = int(slide_align)
        if base_mil == 'abmil':
            from models.abmil import DAttention
            self.mil = DAttention(n_classes=n_classes)
        else:
            raise ValueError(f"Base MIL {base_mil} not supported")

        self.apply(initialize_weights)

        self.encoder_IB = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(input_dim, input_dim * 2),
            ).to("cuda")
        
        if train:

            if task == 'BRCA':

                from models.prompts import brca_prompts
                prompts = brca_prompts()

                visual_prompts_path = './prmpts/brca'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)
                

            elif task == 'BRCA_HER2':

                from models.prompts import brca_her2_prompts
                prompts = brca_her2_prompts()

                visual_prompts_path = './prmpts/brca_her2'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)


            elif task == 'BRCA_ER':

                from models.prompts import brca_er_prompts
                prompts = brca_er_prompts()

                visual_prompts_path = './prmpts/brca_er'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)


            elif task == 'BRCA_PR':

                from models.prompts import brca_pr_prompts
                prompts = brca_pr_prompts()

                visual_prompts_path = './prmpts/brca_pr'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)


            elif task == 'BRCA_PIK3CA':

                from models.prompts import brca_pik3ca_prompts
                prompts = brca_pik3ca_prompts()

                visual_prompts_path = './prmpts/brca_pik3ca'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)


            elif task == 'BRCA_CDH1':

                from models.prompts import brca_cdh1_prompts
                prompts = brca_cdh1_prompts()

                visual_prompts_path = './prmpts/brca_cdh1'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_neg = vis_feats_neg[:8]
                vis_feats_pos = vis_feats_pos[:8]
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)


            elif task == 'LUAD_EGFR':

                from models.prompts import luad_egfr_prompts
                prompts = luad_egfr_prompts()

                visual_prompts_path = './prmpts/luad_egfr'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)


            elif task == 'LUAD_STK11':
                from models.prompts import luad_stk11_prompts
                prompts = luad_stk11_prompts()

                visual_prompts_path = './prmpts/luad_stk11'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            elif task == 'NSCLC':
                from models.prompts import nsclc_prompts
                prompts = nsclc_prompts()
                visual_prompts_path = './prmpts/luad_stk11'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            elif task == 'STAD_MSI':

                from models.prompts import stad_msi_prompts
                prompts = stad_msi_prompts()

                visual_prompts_path = './prmpts/stad_msi'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    # import ipdb; ipdb.set_trace()
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            elif task == 'STAD_EBV':

                from models.prompts import stad_ebv_prompts
                prompts = stad_ebv_prompts()

                visual_prompts_path = './prmpts/stad_ebv'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    # import ipdb; ipdb.set_trace()
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)


            elif task == 'STAD_LAUREN':

                from models.prompts import stad_lauren_prompts
                prompts = stad_lauren_prompts()

                visual_prompts_path = './prmpts/stad_lauren'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/diffuse'):
                    image = Image.open(visual_prompts_path+'/diffuse/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/Intestinal'):
                    image = Image.open(visual_prompts_path+'/Intestinal/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            elif task == 'STAD_TP53':

                from models.prompts import stad_tp53_prompts
                prompts = stad_tp53_prompts()

                visual_prompts_path = './prmpts/stad_tp53'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            elif task == 'STAD_MUC16':

                from models.prompts import stad_muc16_prompts
                prompts = stad_muc16_prompts()

                visual_prompts_path = './prmpts/stad_muc16'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            elif task == 'SARC':
                from models.prompts import sarc_prompts
                prompts = sarc_prompts()


            elif task == 'CRC_MSI':

                from models.prompts import crc_msi_prompts
                prompts = crc_msi_prompts()

                visual_prompts_path = './prmpts/crc_msi'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    # import ipdb; ipdb.set_trace()
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            
            elif task == 'CRC_BRAF':

                from models.prompts import crc_braf_prompts
                prompts = crc_braf_prompts()

                visual_prompts_path = './prmpts/crc_braf'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    # import ipdb; ipdb.set_trace()
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            
            elif task == 'CRC_TP53':

                from models.prompts import crc_tp53_prompts
                prompts = crc_tp53_prompts()

                visual_prompts_path = './prmpts/crc_tp53'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            
            elif task == 'GBM_IDH1':
                from models.prompts import gbm_idh1_prompts
                prompts = gbm_idh1_prompts()
            

            elif task == 'LGG_IDH1':

                from models.prompts import lgg_idh1_prompts
                prompts = lgg_idh1_prompts()

                visual_prompts_path = './prmpts/lgg_idh1'
                vis_feats_pos = []
                vis_feats_neg = []
                for file in os.listdir(visual_prompts_path+'/pos'):
                    image = Image.open(visual_prompts_path+'/pos/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_pos.append(image_embs)
                for file in os.listdir(visual_prompts_path+'/neg'):
                    image = Image.open(visual_prompts_path+'/neg/'+file)
                    image = preprocess(image).unsqueeze(0).to('cuda')
                    with torch.inference_mode():
                        image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
                    vis_feats_neg.append(image_embs)
                vis_feats_neg = vis_feats_neg[:8]
                vis_feats_pos = vis_feats_pos[:8]
                vis_feats_pos = torch.concat(vis_feats_pos, dim=0)
                vis_feats_neg = torch.concat(vis_feats_neg, dim=0)
                self.vis_concepts = torch.stack([vis_feats_neg, vis_feats_pos], dim=0)

            self.ori_feats = []
            for i in range(len(prompts)):
                tokenized_templates = tokenize(texts=prompts[i], tokenizer=get_tokenizer())
                self.ori_feats.append(conch_model.encode_text(tokenized_templates.to('cuda')).detach())
            self.ori_feats = [feat.to("cuda") for feat in self.ori_feats]

            conch_text = conch_model.text
            self.prompt_learner_local = nn.ModuleList([PromptLearner(prompt, conch_text, n_ctx, n_flp, num_patch_prompt, is_shared=False) for prompt in prompts])
            # self.prompt_learner_global = nn.ModuleList([PromptLearner(prompt, conch_text, n_ctx, n_flp, num_patch_prompt, is_shared=False) for prompt in prompts[:2]])
            self.prompt_learner_global = nn.ModuleList([PromptLearner(prompt, conch_text, n_ctx, n_flp, num_patch_prompt, is_shared=False) for prompt in prompts])
            self.tokenized_prompts_local = [learner.tokenized_prompts for learner in self.prompt_learner_local]
            self.tokenized_prompts_global = [learner.tokenized_prompts for learner in self.prompt_learner_global]
            self.text_encoder = TextEncoder(conch_text)
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            self.infonce_loss = InfoNCE()

            # ckpt_path = f'./results/{exp_code}/s1_abmil_conch/s_{fold}_checkpoint.pt'
            # ckpt = torch.load(ckpt_path, map_location='cpu')
            # self.abmil = DAttention(n_classes=n_classes)
            # self.abmil.load_state_dict(ckpt, strict=False)
            # self.abmil = self.abmil.to("cuda")
            # self.abmil.eval()

        # self.criterion = lightly.loss.NTXentLoss()
        # self.criterion = NTXentLoss(temperature=0.1)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def forward(self, x1, x2=None, label=None, return_attn=False,no_norm=False, pretrain=False, club=None):


        x = x1.unsqueeze(0)
        results_dict = {}
        results_dict['ori_x'] = x1.clone().detach()

        x_re = self.encoder_IB(x.squeeze(0))
        mu, logvar = x_re.chunk(2, dim=-1)
        kl_loss = self.kl_loss(mu, logvar)
        x_re = self.reparameterise(mu, logvar)
        k = 100

        results_dict['kl_loss'] = kl_loss

        if label is not None:
            # self.tokenized_prompts_local = [learner.tokenized_prompts for learner in self.prompt_learner_local]
            # self.tokenized_prompts_global = [learner.tokenized_prompts for learner in self.prompt_learner_global]
            # import ipdb; ipdb.set_trace()
            patch_prompts_local = []
            patch_prompts_global = []
            for i in range(len(self.prompt_learner_local)):
                patch_prompts_local.append(self.prompt_learner_local[i]())
            for i in range(len(self.prompt_learner_global)):
                patch_prompts_global.append(self.prompt_learner_global[i]())
            patch_tokenized_prompts_local = self.tokenized_prompts_local
            patch_tokenized_prompts_global = self.tokenized_prompts_global
            self.feats_local = [self.text_encoder(prompt, tokenized_prompt) for prompt, tokenized_prompt in zip(patch_prompts_local, patch_tokenized_prompts_local)]
            self.feats_global = [self.text_encoder(prompt, tokenized_prompt) for prompt, tokenized_prompt in zip(patch_prompts_global, patch_tokenized_prompts_global)]
            
            # sim = x.squeeze(0) @ self.ori_feats[label[0]].mean(dim=0)
            sim = x.squeeze(0) @ self.vis_concepts[label[0]].mean(dim=0)
            # sim = x.squeeze(0) @ (self.vis_concepts[label[0]].mean(dim=0) + self.ori_feats[label[0]].mean(dim=0)) / 2

            h_pos = x_re[torch.topk(sim, k)[1]]

            info_loss = self.infonce_loss(h_pos, repeat(self.feats_local[label[0]].mean(dim=0), 'c -> b c', b=h_pos.shape[0]).detach(), torch.concat([torch.mean(self.feats_local[j], dim=0, keepdim=True) for j in range(len(self.feats_local)) if j != label[0]], dim=0).detach())  # best
            info_loss += self.infonce_loss(self.feats_local[label[0]], repeat(h_pos.mean(dim=0), 'c -> b c', b=self.feats_local[label[0]].shape[0]).detach(), torch.concat([torch.mean(self.feats_local[j], dim=0, keepdim=True) for j in range(len(self.feats_local)) if j != label[0]], dim=0).detach())

            results_dict[f'infonce_loss'] = info_loss
            
        # x = x_re.unsqueeze(0)
        x = x_re

        x, Y_hat, Y_prob, results_dict = self.mil(x, x, label, results_dict=results_dict)
        # results_dict['re_x'] = x_re.clone().detach()

        # feature = self.feature(x)
        # feature = feature.squeeze(0)
        # A = self.attention(feature)

        # A_ori = A.clone().detach()
        # results_dict['A'] = A_ori
        # A = torch.transpose(A, -1, -2)  # KxN
        # A = F.softmax(A, dim=-1)  # softmax over N
        # M = torch.mm(A, feature)  # KxL
        # results_dict['M'] = M

        # add slide alignment
        # print('slide_align', self.slide_align)
        if label is not None and self.slide_align and 'slide_feat' in results_dict:
            slide_feat = results_dict['slide_feat']
            # print('slide_feat', slide_feat.shape)
            slide_info_loss = self.infonce_loss(slide_feat, repeat(self.feats_global[label[0]].mean(dim=0), 'c -> b c', b=slide_feat.shape[0]).detach(), torch.concat([torch.mean(self.feats_global[j], dim=0, keepdim=True) for j in range(len(self.feats_global)) if j != label[0]], dim=0).detach())
            slide_info_loss += self.infonce_loss(self.feats_global[label[0]], repeat(slide_feat.mean(dim=0), 'c -> b c', b=self.feats_global[label[0]].shape[0]).detach(), torch.concat([torch.mean(self.feats_global[j], dim=0, keepdim=True) for j in range(len(self.feats_global)) if j != label[0]], dim=0).detach())
            
            if not pretrain:
                results_dict[f'infonce_loss'] += 1 * slide_info_loss

            # results_dict[f'infonce_loss'] = info_loss

        # x = self.classifier(M)

        # Y_hat = torch.argmax(x)
        # Y_prob = F.softmax(x)
        return x, Y_hat, Y_prob, results_dict