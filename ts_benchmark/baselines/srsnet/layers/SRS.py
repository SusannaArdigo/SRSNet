'''
* @author: EmpyreanMoon
*
* @create: 2025-02-26 16:27
*
* @description: 
'''
import torch.nn as nn
import torch
from einops import rearrange
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SRS(nn.Module):
    """
    Selective Representation Space block (Wu et al., NeurIPS 2025).

    Pipeline per batch (input arrives already RevIN-normalized by SRSNetModel):
        1. Pad the input on the right so the last patch fits exactly.
        2. ORIGINAL view  -> adjacent patching with stride         (conventional)
        3. RECONSTRUCTIVE view -> stride-1 candidates + select + shuffle
        4. Embed both views with two separate Linear projections
        5. Adaptive Fusion: learned convex combination via sigmoid(alpha)
        6. Add positional embedding, dropout, return
    """

    def __init__(self, d_model, patch_len, stride, seq_len, dropout, hidden_size, alpha=2.0, pos=True):
        super(SRS, self).__init__()

        # --- Patching geometry ------------------------------------------------
        self.patch_len = patch_len                                                                  # how long each patch is
        self.stride = stride                                                                        # step between adjacent patches
        self.seq_len = seq_len                                                                      # length of the input window

        # Conventional adjacent patching produces exactly patch_num patches
        self.patch_num = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1               # = ceil((T - p) / s) + 1
        # Right-padding so the last adjacent patch fits without overflow
        self.padding = self.patch_len + (self.patch_num - 1) * self.stride - self.seq_len           # extra samples needed
        # Replication padding: repeats the last value (paper-compatible)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))                           # pad only on the right

        # --- Selective Patching scorer^s (paper Sec. 3.2) --------------------
        # For each candidate patch (stride=1), output patch_num scores
        # → one score per output "slot"; we'll argmax along the candidate axis
        self.scorer_select = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size), nn.ReLU(),                                      # patch_len -> hidden
            nn.Linear(hidden_size, self.patch_num)                                                  # hidden -> patch_num scores
        )

        # --- Dynamic Reassembly scorer^r (paper Sec. 3.3) --------------------
        # ONE score per already-selected patch → we'll argsort to get the new order
        self.scorer_shuffle = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size), nn.ReLU(),                                      # patch_len -> hidden
            nn.Linear(hidden_size, 1)                                                               # hidden -> 1 score
        )

        # --- Patch embeddings: two parallel linear projections ----------------
        # The Adaptive Fusion (Sec. 3.4) needs TWO separate embeddings, one per view
        self.value_embedding_org = nn.Linear(patch_len, d_model, bias=False)                        # PatchEmbedding^1  (eq. 12)
        self.value_embedding_rec = nn.Linear(patch_len, d_model, bias=False)                        # PatchEmbedding^2  (eq. 12)

        # --- Positional embedding (sinusoidal, Vaswani et al. 2017) ----------
        if pos:
            self.position_embedding = PositionalEmbedding(d_model)
        self.pos = pos

        # Final dropout on the fused embedding
        self.dropout = nn.Dropout(dropout)

        # --- Adaptive Fusion weight (alpha in eq. 13) ------------------------
        # Shape (patch_num, d_model) → learned per-position, per-channel weight
        # Init at 2.0 → sigmoid(2.0) ≈ 0.88 ⇒ at start we trust the original view
        self.alpha = nn.Parameter(torch.ones(self.patch_num, d_model) * alpha)

    def _origin_view(self, x):
        """Conventional adjacent patching (the 'standard' view)."""
        # x: [B, N, T_padded] → unfold the time axis with stride
        x_origin = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)                    # [B, N, patch_num, patch_len]
        # Merge batch and channel: each channel processed independently
        origin_patches = rearrange(x_origin, 'b c n p -> (b c) n p')                                # [B*N, patch_num, patch_len]
        return origin_patches

    def _rec_view(self, x):
        """Reconstructive view: select n patches from K candidates + reorder them."""
        # All possible candidate patches with stride=1: K = T_padded - patch_len + 1
        x_rec = x.unfold(dimension=-1, size=self.patch_len, step=1)                                 # [B, N, K, patch_len]
        # Pick patch_num patches from the K candidates (with replacement allowed)
        selected_patches = self._select(x_rec)                                                      # [B, N, patch_num, patch_len]
        # Reorder the selected patches via a learned argsort
        shuffled_patches = self._shuffle(selected_patches)                                          # [B, N, patch_num, patch_len]
        # Same flatten as _origin_view (channel-independent)
        rec_patches = rearrange(shuffled_patches, 'b c n p -> (b c) n p')                           # [B*N, patch_num, patch_len]
        return rec_patches

    def _select(self, x_rec):
        """Selective Patching: pick the best candidate for each output slot."""
        # x_rec: [B, N, K, patch_len] → score every candidate, output patch_num scores each
        scores = self.scorer_select(x_rec)                                                          # [B, N, K, patch_num]
        # For each output slot, take the candidate with the highest score (argmax along K)
        indices = torch.argmax(scores, dim=-2, keepdim=True)                                        # [B, N, 1, patch_num]
        # Gather the max scores → we need them for the straight-through gradient trick (eq. 3-5)
        max_scores = torch.gather(input=scores, dim=-2, index=indices)                              # [B, N, 1, patch_num]
        non_zero_mask = max_scores != 0
        # Detached reciprocal: only max_scores carries the gradient
        inv = (1 / max_scores[non_zero_mask]).detach()                                              # stop-grad on 1/score

        # Expand indices along patch_len so we can gather the actual patch values
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)                 # [B, N, patch_num, patch_len]
        # Pull the selected patches out of the candidate tensor
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)                   # [B, N, patch_num, patch_len]

        # Straight-through estimator: value = patches * (max_score / max_score)_detach
        # → numerically identity, but gradient flows back through max_score → scorer_select
        max_scores[non_zero_mask] *= inv
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches                        # [B, N, patch_num, patch_len]

        return selected_patches

    def _shuffle(self, selected_patches):
        """Dynamic Reassembly: argsort the selected patches by a learned score."""
        # selected_patches: [B, N, patch_num, patch_len] → ONE score per patch
        shuffle_scores = self.scorer_shuffle(selected_patches)                                      # [B, N, patch_num, 1]
        # Sort indices by score (descending) → new permutation of the patch axis
        shuffle_indices = torch.argsort(input=shuffle_scores, dim=-2, descending=True)              # [B, N, patch_num, 1]
        # Gather the scores in sorted order
        shuffled_scores = torch.gather(input=shuffle_scores, index=shuffle_indices, dim=-2)         # [B, N, patch_num, 1]
        non_zero_mask = shuffled_scores != 0
        # Same detached reciprocal trick as _select
        inv = (1 / shuffled_scores[non_zero_mask]).detach()

        # Broadcast the indices on the patch_len axis to actually reorder the patches
        shuffle_patch_indices = shuffle_indices.repeat(1, 1, 1, self.patch_len)                     # [B, N, patch_num, patch_len]
        shuffled_patches = torch.gather(input=selected_patches, index=shuffle_patch_indices, dim=-2)# [B, N, patch_num, patch_len]
        # Straight-through estimator (eq. 8-10): identity in value, gradient via shuffled_scores
        shuffled_scores[non_zero_mask] *= inv
        shuffled_patches = shuffled_scores * shuffled_patches                                       # [B, N, patch_num, patch_len]

        return shuffled_patches

    def forward(self, x):
        # x: [B, N, T] -- channel-independent, already RevIN-normalized upstream by SRSNetModel
        n_vars = x.shape[1]                                                                         # number of channels (N)
        # Pad on the right so the last adjacent patch fits exactly
        x = self.padding_patch_layer(x)                                                             # [B, N, T_padded]

        # === Reconstructive view (selective patching + dynamic reassembly) ===
        rec_repr_space = self._rec_view(x)                                                          # [B*N, patch_num, patch_len]

        # === Original view (conventional adjacent patching) ===
        original_repr_space = self._origin_view(x)                                                  # [B*N, patch_num, patch_len]

        # === Adaptive Fusion (paper eq. 13) ===
        weight = torch.sigmoid(self.alpha)                                                          # alpha -> (0,1)^(patch_num, d_model)
        # Convex combination of the two embedded views:
        #     E_tilde = alpha * E_original + (1 - alpha) * E_selective
        embedding = weight * self.value_embedding_org(original_repr_space) \
                    + (1 - weight) * self.value_embedding_rec(rec_repr_space)                       # [B*N, patch_num, d_model]

        # === Positional embedding (paper eq. 14) ===
        if self.pos:
            position_embedding = self.position_embedding(original_repr_space)                       # [1, patch_num, d_model]
            embedding = embedding + position_embedding

        # Final dropout for regularization
        return self.dropout(embedding), n_vars
