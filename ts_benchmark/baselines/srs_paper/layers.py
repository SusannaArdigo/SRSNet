
import math

import torch
from einops import rearrange
from torch import nn

from ts_benchmark.baselines.srsnet.layers.SRS import PositionalEmbedding, SRS


class SRSNoSRS(SRS):

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)                          
        original_repr_space = self._origin_view(x)               
        embedding = self.value_embedding_org(original_repr_space)
        if self.pos:
            embedding = embedding + self.position_embedding(original_repr_space)
        return self.dropout(embedding), n_vars


class SRSNoSelectivePatching(SRS):

    def _rec_view(self, x):
        original = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  
        shuffled_patches = self._shuffle(original)                                
        return rearrange(shuffled_patches, "b c n p -> (b c) n p")


class SRSNoDynamicReassembly(SRS):

    def _rec_view(self, x):
        x_rec = x.unfold(dimension=-1, size=self.patch_len, step=1)   
        selected_patches = self._select(x_rec)                        
        return rearrange(selected_patches, "b c n p -> (b c) n p")


class SRSNoAdaptiveFusion(SRS):

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        rec_repr_space = self._rec_view(x)                          
        original_repr_space = self._origin_view(x)                 

        embedding = 0.5 * self.value_embedding_org(original_repr_space)
        embedding = embedding + 0.5 * self.value_embedding_rec(rec_repr_space)
        if self.pos:
            embedding = embedding + self.position_embedding(original_repr_space)
        return self.dropout(embedding), n_vars


class SRSAsPatchEmbedding(nn.Module):

    def __init__(
        self,
        d_model,
        patch_len,
        stride,
        padding,
        dropout,
        hidden_size=128,
        alpha=2.0,
        pos=True,
        seq_len=None,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        self.patch_num = None
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.scorer_select = None
        self.scorer_shuffle = None
        self.value_embedding_org = nn.Linear(patch_len, d_model, bias=False)
        self.value_embedding_rec = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model) if pos else None
        self.pos = pos
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.alpha_init = alpha
        self.alpha = None
        if seq_len is not None:
            self._ensure_shape(patch_count(seq_len, patch_len, stride, padding), torch.device("cpu"))

    def _ensure_shape(self, patch_num, device):

        if self.patch_num == patch_num:
            return
        self.patch_num = patch_num
        self.scorer_select = nn.Sequential(                                
            nn.Linear(self.patch_len, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, patch_num),
        ).to(device)
        self.scorer_shuffle = nn.Sequential(                              
            nn.Linear(self.patch_len, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        ).to(device)
        alpha = torch.ones(patch_num, self.value_embedding_org.out_features, device=device)
        self.alpha = nn.Parameter(alpha * self.alpha_init)                 

    def _select(self, x_rec):
        scores = self.scorer_select(x_rec)
        indices = torch.argmax(scores, dim=-2, keepdim=True)
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)
        max_scores[non_zero_mask] *= inv
        return max_scores.permute(0, 1, 3, 2) * selected_patches

    def _shuffle(self, selected_patches):
        shuffle_scores = self.scorer_shuffle(selected_patches)
        shuffle_indices = torch.argsort(input=shuffle_scores, dim=-2, descending=True)
        shuffled_scores = torch.gather(input=shuffle_scores, index=shuffle_indices, dim=-2)
        non_zero_mask = shuffled_scores != 0
        inv = (1 / shuffled_scores[non_zero_mask]).detach()
        shuffle_patch_indices = shuffle_indices.repeat(1, 1, 1, self.patch_len)
        shuffled_patches = torch.gather(input=selected_patches, index=shuffle_patch_indices, dim=-2)
        shuffled_scores[non_zero_mask] *= inv
        return shuffled_scores * shuffled_patches

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        original = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   
        patch_num = original.shape[-2]
        self._ensure_shape(patch_num, x.device)                                   
        candidates = x.unfold(dimension=-1, size=self.patch_len, step=1)          
        rec = self._shuffle(self._select(candidates))                              
        original = rearrange(original, "b c n p -> (b c) n p")
        rec = rearrange(rec, "b c n p -> (b c) n p")
        weight = torch.sigmoid(self.alpha)
        embedding = weight * self.value_embedding_org(original)
        embedding = embedding + (1 - weight) * self.value_embedding_rec(rec)
        if self.position_embedding is not None:
            embedding = embedding + self.position_embedding(original)
        return self.dropout(embedding), n_vars


def patch_count(seq_len, patch_len, stride, padding=0):
    return math.floor((seq_len + padding - patch_len) / stride) + 1
