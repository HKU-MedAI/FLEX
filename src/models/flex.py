import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import repeat
import os
from PIL import Image
from PIL import PngImagePlugin

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from models.infonce import InfoNCE
from models.conch_custom import TextEncoder, PromptLearner

# Increase chunk size for large PNG files
PngImagePlugin.MAX_TEXT_CHUNK = 1024*2**20

# Initialize CONCH model
conch_model, preprocess = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path="./models/weights/conch.bin")
conch_model = conch_model.to("cuda")


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


class FLEX(nn.Module):
    """FLEX model for WSI classification using vision-language representations"""
    
    def __init__(self, input_dim=512, n_classes=2, dropout=0.25, act='relu', 
                 rrt=None, n_ctx=None, n_flp=0, num_patch_prompt=0, 
                 task=None, fold=None, exp_code=None, train=True, 
                 base_mil=None, slide_align=1):
        super(FLEX, self).__init__()
        self.slide_align = int(slide_align)
        
        # Initialize MIL model
        if base_mil == 'abmil':
            from models.abmil import DAttention
            self.mil = DAttention(n_classes=n_classes)
        else:
            raise ValueError(f"Base MIL {base_mil} not supported")

        self.apply(initialize_weights)

        # Information bottleneck encoder
        self.encoder_IB = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim * 2),
        ).to("cuda")
        
        if train:
            # Load task-specific prompts and visual concepts
            prompts, self.vis_concepts = self._load_task_data(task)
            
            # Initialize text encoders and prompt learners
            self._init_text_encoders(prompts, n_ctx, n_flp, num_patch_prompt)
            
            # Initialize InfoNCE loss
            self.infonce_loss = InfoNCE()

    def _load_task_data(self, task):
        """Load task-specific prompts and visual concepts"""
        prompts = None
        vis_concepts = self._load_visual_prompts(f'./prompts/{task}')
        
        if task == 'BRCA':
            from models.prompts import brca_prompts
            prompts = brca_prompts()
            
        elif task == 'BRCA_HER2':
            from models.prompts import brca_her2_prompts
            prompts = brca_her2_prompts()
            
        elif task == 'BRCA_ER':
            from models.prompts import brca_er_prompts
            prompts = brca_er_prompts()
            
        elif task == 'BRCA_PR':
            from models.prompts import brca_pr_prompts
            prompts = brca_pr_prompts()
            
        elif task == 'BRCA_PIK3CA':
            from models.prompts import brca_pik3ca_prompts
            prompts = brca_pik3ca_prompts()
            
        elif task == 'BRCA_CDH1':
            from models.prompts import brca_cdh1_prompts
            prompts = brca_cdh1_prompts()
            
        elif task == 'LUAD_EGFR':
            from models.prompts import luad_egfr_prompts
            prompts = luad_egfr_prompts()
            
        elif task == 'LUAD_STK11':
            from models.prompts import luad_stk11_prompts
            prompts = luad_stk11_prompts()
            
        elif task == 'NSCLC':
            from models.prompts import nsclc_prompts
            prompts = nsclc_prompts()
            
        elif task == 'STAD_MSI':
            from models.prompts import stad_msi_prompts
            prompts = stad_msi_prompts()
            
        elif task == 'STAD_EBV':
            from models.prompts import stad_ebv_prompts
            prompts = stad_ebv_prompts()
            
        elif task == 'STAD_LAUREN':
            from models.prompts import stad_lauren_prompts
            prompts = stad_lauren_prompts()
            
        elif task == 'STAD_TP53':
            from models.prompts import stad_tp53_prompts
            prompts = stad_tp53_prompts()
            
        elif task == 'STAD_MUC16':
            from models.prompts import stad_muc16_prompts
            prompts = stad_muc16_prompts()
            
        elif task == 'CRC_BRAF':
            from models.prompts import crc_braf_prompts
            prompts = crc_braf_prompts()
            
        elif task == 'CRC_TP53':
            from models.prompts import crc_tp53_prompts
            prompts = crc_tp53_prompts()
            
        return prompts, vis_concepts

    def _load_visual_prompts(self, base_path, max_samples=None):
        """Load visual prompts from directory"""
        vis_feats_1 = []
        vis_feats_0 = []
        
        # Load positive samples
        pos_files = os.listdir(f"{base_path}/1")
        if max_samples:
            pos_files = pos_files[:max_samples]
            
        for file in pos_files:
            image = Image.open(f"{base_path}/1/{file}")
            image = preprocess(image).unsqueeze(0).to('cuda')
            with torch.inference_mode():
                image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
            vis_feats_1.append(image_embs)
            
        # Load negative samples
        neg_files = os.listdir(f"{base_path}/0")
        if max_samples:
            neg_files = neg_files[:max_samples]
            
        for file in neg_files:
            image = Image.open(f"{base_path}/0/{file}")
            image = preprocess(image).unsqueeze(0).to('cuda')
            with torch.inference_mode():
                image_embs = conch_model.encode_image(image, proj_contrast=True, normalize=True)
            vis_feats_0.append(image_embs)
            
        vis_feats_1 = torch.concat(vis_feats_1, dim=0)
        vis_feats_0 = torch.concat(vis_feats_0, dim=0)
        
        return torch.stack([vis_feats_0, vis_feats_1], dim=0)

    
    def _init_text_encoders(self, prompts, n_ctx, n_flp, num_patch_prompt):
        """Initialize text encoders and prompt learners"""
        self.ori_feats = []
        
        # Generate text features
        for i in range(len(prompts)):
            tokenized_templates = tokenize(texts=prompts[i], tokenizer=get_tokenizer())
            self.ori_feats.append(conch_model.encode_text(tokenized_templates.to('cuda')).detach())
        
        self.ori_feats = [feat.to("cuda") for feat in self.ori_feats]

        # Initialize prompt learners
        conch_text = conch_model.text
        self.prompt_learner_local = nn.ModuleList([
            PromptLearner(prompt, conch_text, n_ctx, n_flp, num_patch_prompt, is_shared=False) 
            for prompt in prompts
        ])
        
        self.prompt_learner_global = nn.ModuleList([
            PromptLearner(prompt, conch_text, n_ctx, n_flp, num_patch_prompt, is_shared=False) 
            for prompt in prompts
        ])
        
        self.tokenized_prompts_local = [learner.tokenized_prompts for learner in self.prompt_learner_local]
        self.tokenized_prompts_global = [learner.tokenized_prompts for learner in self.prompt_learner_global]
        
        # Initialize text encoder
        self.text_encoder = TextEncoder(conch_text)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def reparameterise(self, mu, logvar):
        """Reparameterization trick for variational autoencoder"""
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def kl_loss(self, mu, logvar):
        """KL divergence loss for variational autoencoder"""
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return torch.mean(kl_div)

    def forward(self, x1, x2=None, label=None):
        """Forward pass through the model"""
        x = x1.unsqueeze(0)
        results_dict = {'ori_x': x1.clone().detach()}

        # Encode through information bottleneck
        x_re = self.encoder_IB(x.squeeze(0))
        mu, logvar = x_re.chunk(2, dim=-1)
        kl_loss = self.kl_loss(mu, logvar)
        x_re = self.reparameterise(mu, logvar)
        results_dict['kl_loss'] = kl_loss

        # Process text features if label is provided
        if label is not None and hasattr(self, 'prompt_learner_local'):
            # Generate patch prompts
            patch_prompts_local = [self.prompt_learner_local[i]() for i in range(len(self.prompt_learner_local))]
            patch_prompts_global = [self.prompt_learner_global[i]() for i in range(len(self.prompt_learner_global))]
            
            # Get text features
            self.feats_local = [
                self.text_encoder(prompt, tokenized_prompt) 
                for prompt, tokenized_prompt in zip(patch_prompts_local, self.tokenized_prompts_local)
            ]
            
            self.feats_global = [
                self.text_encoder(prompt, tokenized_prompt) 
                for prompt, tokenized_prompt in zip(patch_prompts_global, self.tokenized_prompts_global)
            ]
            
            # Calculate similarity and get top k features
            k = 100
            sim = x.squeeze(0) @ self.vis_concepts[label[0]].mean(dim=0)
            h_pos = x_re[torch.topk(sim, k)[1]]

            # Calculate InfoNCE loss
            info_loss = self.infonce_loss(
                h_pos, 
                repeat(self.feats_local[label[0]].mean(dim=0), 'c -> b c', b=h_pos.shape[0]).detach(), 
                torch.concat([
                    torch.mean(self.feats_local[j], dim=0, keepdim=True) 
                    for j in range(len(self.feats_local)) if j != label[0]
                ], dim=0).detach()
            )
            
            info_loss += self.infonce_loss(
                self.feats_local[label[0]], 
                repeat(h_pos.mean(dim=0), 'c -> b c', b=self.feats_local[label[0]].shape[0]).detach(), 
                torch.concat([
                    torch.mean(self.feats_local[j], dim=0, keepdim=True) 
                    for j in range(len(self.feats_local)) if j != label[0]
                ], dim=0).detach()
            )

            results_dict['infonce_loss'] = info_loss
            
        # Forward through MIL model
        x = x_re
        x, Y_hat, Y_prob, results_dict = self.mil(x, x, label, results_dict=results_dict)

        # Apply slide alignment if needed
        if label is not None and self.slide_align and 'slide_feat' in results_dict and hasattr(self, 'feats_global'):
            slide_feat = results_dict['slide_feat']
            
            slide_info_loss = self.infonce_loss(
                slide_feat, 
                repeat(self.feats_global[label[0]].mean(dim=0), 'c -> b c', b=slide_feat.shape[0]).detach(), 
                torch.concat([
                    torch.mean(self.feats_global[j], dim=0, keepdim=True) 
                    for j in range(len(self.feats_global)) if j != label[0]
                ], dim=0).detach()
            )
            
            slide_info_loss += self.infonce_loss(
                self.feats_global[label[0]], 
                repeat(slide_feat.mean(dim=0), 'c -> b c', b=self.feats_global[label[0]].shape[0]).detach(), 
                torch.concat([
                    torch.mean(self.feats_global[j], dim=0, keepdim=True) 
                    for j in range(len(self.feats_global)) if j != label[0]
                ], dim=0).detach()
            )
            
            results_dict['infonce_loss'] += 1 * slide_info_loss

        return x, Y_hat, Y_prob, results_dict