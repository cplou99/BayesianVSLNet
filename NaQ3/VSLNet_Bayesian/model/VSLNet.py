"""VSLNet Baseline for Ego4D Episodic Memory -- Natural Language Queries.
"""
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from model.layers import (
    Embedding,
    VisualProjection,
    OurScaledVisualProjection,
    FeatureEncoder,
    CQAttention,
    CQConcatenate,
    ConditionedPredictor,
    OneHotConditionedPredictor,
    HighLightLayer,
    BertEmbedding,
)

import sys
#sys.path.append('/home/ego_exo4d/TAS')
print(sys.path)
from model.EgoVLP_text_model import TextModel_EgoVLP


def build_optimizer_and_scheduler(model, configs):
    no_decay = [
        "bias",
        "layer_norm",
        "LayerNorm",
    ]  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        configs.num_train_steps * configs.warmup_proportion,
        configs.num_train_steps,
    )
    return optimizer, scheduler


class VSLNet(nn.Module):
    def __init__(self, configs, word_vectors):
        super(VSLNet, self).__init__()
        self.configs = configs
        if configs.gradual_mlp:
            self.video_affine = OurScaledVisualProjection(
                visual_dim=configs.video_feature_dim,
                dim=configs.dim,
                drop_rate=configs.drop_rate,
            )
        else:
            self.video_affine = VisualProjection(
                visual_dim=configs.video_feature_dim,
                dim=configs.dim,
                drop_rate=configs.drop_rate,
            )
        
        self.feature_encoder = FeatureEncoder(
            dim=configs.dim,
            num_heads=configs.num_heads,
            kernel_size=7,
            num_layers=4,
            max_pos_len=configs.max_pos_len,
            drop_rate=configs.drop_rate,
            num_extra_attn_blocks=configs.num_extra_attn_blocks,
        )
        # video and query fusion
        self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        self.cq_concat = CQConcatenate(dim=configs.dim)
        # query-guided highlighting
        self.highlight_layer = HighLightLayer(dim=configs.dim)
        # conditioned predictor
        # self.predictor = ConditionedPredictor(
        #    dim=configs.dim,
        #    num_heads=configs.num_heads,
        #    drop_rate=configs.drop_rate,
        #    max_pos_len=configs.max_pos_len,
        #    predictor=configs.predictor,
        #)

        self.new_h_predictor = OneHotConditionedPredictor(
            dim=configs.dim,
            num_heads=configs.num_heads,
            drop_rate=configs.drop_rate,
            max_pos_len=configs.max_pos_len,
            predictor=configs.predictor,
            return_sigmoid=False,
        )

        self.bce = nn.BCELoss(reduction="mean")

        # If pretrained transformer, initialize_parameters and load.
        if configs.predictor == "bert":
            # Project back from BERT to dim.
            self.query_affine = nn.Linear(768, configs.dim)
            # init parameters
            self.init_parameters()
            self.embedding_net = BertEmbedding(configs.text_agnostic)
        elif configs.predictor == "RObertA_EgoVLPv2":
            print("Loading weights from Roberta")
            self.query_affine = nn.Linear(768, configs.dim) 
            self.init_parameters()
            self.embedding_net = TextModel_EgoVLP()
            model_weights = self.embedding_net.model_weights
            self.embedding_net.load_state_dict(model_weights, strict=True)
            for param in self.embedding_net.parameters():
                param.requires_grad = False
        else:
            self.embedding_net = Embedding(
                num_words=configs.word_size,
                num_chars=configs.char_size,
                out_dim=configs.dim,
                word_dim=configs.word_dim,
                char_dim=configs.char_dim,
                word_vectors=word_vectors,
                drop_rate=configs.drop_rate,
            )
            # init parameters
            self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Linear)
            ):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def forward(
        self,
        word_ids,
        char_ids,
        video_features,
        v_mask,
        q_mask,
        get_losses=False,
        labels=None,
    ):
        video_features = self.video_affine(video_features)
        if self.configs.predictor == "bert":
            query_features = self.embedding_net(word_ids)
            query_features = self.query_affine(query_features)
        elif self.configs.predictor == "RObertA_EgoVLPv2":
            #Now, we compute text features WIHOUT EGOVLP HEAD -> THE OUTPUT DIMENSION IS 768
            query_features = self.embedding_net.compute_text(word_ids).contiguous()
            query_features = self.query_affine(query_features)
        else:
            query_features = self.embedding_net(word_ids, char_ids)

        query_features = self.feature_encoder(query_features, mask=q_mask)
        video_features = self.feature_encoder(video_features, mask=v_mask)
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)
        logits = self.new_h_predictor(features, v_mask)
    
        if get_losses:
            assert labels is not None
            loss = self.compute_probs_loss(logits, labels["h_labels"], v_mask)
            probs = torch.sigmoid(logits)
            return probs, loss
        else:
            return torch.sigmoid(logits)


    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(
            start_logits=start_logits, end_logits=end_logits
        )

    def extract_index_from_probs(self, probs, num_times=1, num_text_queries=1, topk=1):
        return self.new_h_predictor.extract_index(
            probs=probs, num_times=num_times, num_text_queries=num_text_queries, topk=topk
        )
    
    def compute_probs_loss(self, probs, labels, mask):
        probs = probs.float()
        labels = labels.float()
        return self.new_h_predictor.compute_loss(
               scores=probs, labels=labels, mask=mask
        )
    
    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(
            scores=scores, labels=labels, mask=mask
        )

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
        )
