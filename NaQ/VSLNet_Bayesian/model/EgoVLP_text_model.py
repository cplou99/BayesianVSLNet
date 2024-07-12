# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb

import timm
import torch
import yaml
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel
from einops import rearrange, repeat


from model.roberta import RobertaModel, _prepare_decoder_attention_mask
from transformers import RobertaConfig
from functools import partial
import copy
import torch.distributed as dist
import model.parse_config


import torch.nn as nn
import numpy as np
from abc import abstractmethod
import json


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters)
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


root_dir = os.path.dirname(os.path.dirname(os.getcwd()))

with open(f'{root_dir}/NaQ/VSLNet_Bayesian/model/EgoNCE_MLM_ITM_Config.yml') as f:
    config_yaml = yaml.load(f, Loader=yaml.FullLoader)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

f = open(f'{root_dir}/NaQ/VSLNet_Bayesian/model/egovlp_config.json')
config_json = json.load(f)
text_params = {"model": config_json['arch']['args']['text_params']['model'], "pretrained": True, "input": config_json['arch']['args']['text_params']['input']}

class TextModel_EgoVLP(BaseModel):
    def __init__(self,
                 #video_params,
                 text_params = text_params,
                 projection_dim=4096,
                 load_checkpoint=f'{root_dir}/NaQ/VSLNet_Bayesian/model/EgoVLP_weights',
                 projection='minimal',
                 load_temporal_fix='bilinear',
                 config = config_yaml,
                 task_names = 'EgoNCE_ITM_MLM',
                 norm_layer = None,
                 embed_dim=768):
        super().__init__()

        self.text_params = text_params
        self.load_temporal_fix = load_temporal_fix
        self.config = config
        self.task_names = task_names
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        if self.text_params['model'].startswith('roberta'):
            self.text_model = RobertaModel.from_pretrained("roberta-base")
        self.text_model.train()
        
        self.text_model_weights = os.path.join(load_checkpoint, 'EgoVLP_text_model.pth')
        self.txt_proj_weights = os.path.join(load_checkpoint, 'EgoVLP_txt_proj.pth')
        

        # Project to a common embedding
        if projection == 'minimal':

            txt_proj = nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, projection_dim, bias=False),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True),
                nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim, bias=True)
            )

        elif projection == '':
            txt_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj

        if load_checkpoint not in ["", None]:
            print(f"Loading checkpoint from {load_checkpoint}", '**********************************')
            """
            text_model_weights = {}
            txt_proj_weights = {}
            vid_proj_weights = {}
            checkpoint = torch.load(load_checkpoint, map_location='cpu')
            
            state_dict = checkpoint['state_dict']
            for k in state_dict.keys():
                print(k, state_dict[k].shape)
                if k.startswith('module.text_model'):
                    text_model_weights[k[7:] ] = state_dict[k]
                elif k.startswith('module.txt_proj'):
                    txt_proj_weights[k[7:]] = state_dict[k]
                elif k.startswith('module.vid_proj'):
                    vid_proj_weights[k[7:]] = state_dict[k]
            torch.save(text_model_weights, '/home/ego_exo4d/TAS/NaQ/VSLNet/model/EgoVLP_weights/EgoVLP_text_model.pth')
            torch.save(txt_proj_weights, '/home/ego_exo4d/TAS/NaQ/VSLNet/model/EgoVLP_weights/EgoVLP_txt_proj.pth')
            torch.save(vid_proj_weights, '/home/ego_exo4d/TAS/NaQ/VSLNet/model/EgoVLP_weights/EgoVLP_vid_proj.pth')
            """
            text_model_weights = torch.load(self.text_model_weights, map_location='cpu')
            txt_proj_weights = torch.load(self.txt_proj_weights, map_location='cpu')
            self.model_weights = {**text_model_weights, **txt_proj_weights}
            print('text_model_weights', txt_proj_weights.keys())
            
            

    def set_device(self, device):
        self.device = device


    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        elif self.text_params['model'].startswith('roberta'):
            # Input: (32, 14) -> Output: (32, 768)
            # Input: (32, 14) -> Output: (32, 14, 768)
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError
        # WE DO NOT USE THE PROJECTION HEAD
        #if self.config['use_checkpoint']:
        #    text_embeddings = torch.utils.checkpoint.checkpoint(self.txt_proj, text_embeddings)
        #else:
        #    text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']    # not implement for bert
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        elif self.text_params['model'].startswith('roberta'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError

        if self.config['use_checkpoint']:
            text_embeddings = torch.utils.checkpoint.checkpoint(self.txt_proj, text_embeddings)
        else:
            text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def sim_matrix_batch_val(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=-1).unsqueeze(-1), b.norm(dim=-1).unsqueeze(-1)
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


if __name__ == "__main__":
    #pass
    f = open(f'{root_dir}/NaQ/VSLNet_Bayesian/model/egovlp_config.json')
    config = json.load(f)
    text_params = {"model": config['arch']['args']['text_params']['model'], "pretrained": True, "input": config['arch']['args']['text_params']['input']}
    model = TextModel_EgoVLP(text_params)
    state_dict = model.model_weights
    model.load_state_dict(state_dict, strict=True)
    for name, param in model.named_parameters():
        print(name, param.size())
