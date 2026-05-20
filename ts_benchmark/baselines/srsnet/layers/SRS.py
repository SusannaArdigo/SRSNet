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

    Paper equations are referenced inline (e.g. `# eq. 2:`) right above each
    implementation line. 
    Notation: 
        [ ] N=channels,
        [ ] T=lookback,
        [ ] p=patch_len,
        [ ] s=stride,
        [ ] n=patch_num,
        [ ] K=(n-1)*s+1 candidates,
        [ ] d=d_model.
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
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))                           # [B, N, T] -> [B, N, T+padding]

        # --- Selective Patching scorer^s (paper Sec. 3.2) --------------------
        # For each candidate patch (stride=1), output patch_num scores
        # → one score per output "slot"; we'll argmax along the candidate axis
        # Module maps:  [B, N, K, p] -> [B, N, K, n]
        self.scorer_select = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size), nn.ReLU(),                                      # [B, N, K, p] -> [B, N, K, h]
            nn.Linear(hidden_size, self.patch_num)                                                  # [B, N, K, h] -> [B, N, K, n]
        )

        # --- Dynamic Reassembly scorer^r (paper Sec. 3.3) --------------------
        # ONE score per already-selected patch → we'll argsort to get the new order
        # Module maps:  [B, N, n, p] -> [B, N, n, 1]
        self.scorer_shuffle = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size), nn.ReLU(),                                      # [B, N, n, p] -> [B, N, n, h]
            nn.Linear(hidden_size, 1)                                                               # [B, N, n, h] -> [B, N, n, 1]
        )

        # --- Patch embeddings: two parallel linear projections ----------------
        # The Adaptive Fusion (Sec. 3.4) needs TWO separate embeddings, one per view
        self.value_embedding_org = nn.Linear(patch_len, d_model, bias=False)                        # [..., p] -> [..., d_model]  PatchEmbedding^1 (eq. 12)
        self.value_embedding_rec = nn.Linear(patch_len, d_model, bias=False)                        # [..., p] -> [..., d_model]  PatchEmbedding^2 (eq. 12)

        # --- Positional embedding (sinusoidal, Vaswani et al. 2017) ----------
        if pos:
            self.position_embedding = PositionalEmbedding(d_model)                                  # buffer pe: [1, max_len, d_model]
        self.pos = pos                                                                              # bool   -- enable / disable PE

        # Final dropout on the fused embedding
        self.dropout = nn.Dropout(dropout)                                                          # float prob -- no shape change

        # --- Adaptive Fusion weight (alpha in eq. 13) ------------------------
        # Init at 2.0 → sigmoid(2.0) ≈ 0.88 ⇒ at start we trust the original view
        self.alpha = nn.Parameter(torch.ones(self.patch_num, d_model) * alpha)                      # [n, d_model]  -- learnable per-position, per-channel weight

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
        """Selective Patching (paper Sec. 3.2): pick the best candidate for each output slot."""
        # x_rec: [B, N, K, p]  =  P' in the paper (all candidate patches with stride=1)

        # eq. 1 + 2:  S^s = Scorer^s(P'),  I^s = Argmax(S^s)
        scores = self.scorer_select(x_rec)                                  # [B, N, K, n]   -- score every candidate (n scores each)
        indices = torch.argmax(scores, dim=-2, keepdim=True)                # [B, N, 1, n]   -- for each slot pick the top-scoring candidate

        # eq. 3:  S^s_max = S^s[I^s],  S^s_inv = detach(1/S^s_max)
        max_scores = torch.gather(input=scores, dim=-2, index=indices)      # [B, N, 1, n]   -- pull out the winning scores
        non_zero_mask = max_scores != 0                                     # [B, N, 1, n]   -- True where score != 0
        inv = (1 / max_scores[non_zero_mask]).detach()                      # 1D flat        -- detached reciprocal (no grad through it)

        # eq. 4 part 1:  P^s_max = P'[I^s]
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)   # [B, N, n, p] -- expand indices along patch_len
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)     # [B, N, n, p] -- gather the n winning patches

        # eq. 4 part 2 + eq. 5:  E^s = S^s_max ⊙ S^s_inv,  P~^s_max = P^s_max ⊙ E^s
        # Straight-through trick: max_scores * inv == 1 numerically,
        # but the gradient still flows through max_scores -> scorer_select.
        max_scores[non_zero_mask] *= inv                                              # in-place    -- build E^s (value = 1)
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches          # [B, N, n, p] -- multiply by "1" → bridges the gradient

        return selected_patches

    def _shuffle(self, selected_patches):
        """Dynamic Reassembly (paper Sec. 3.3): reorder the selected patches by a learned score."""
        # selected_patches: [B, N, n, p]  =  P~^s_max (output of Selective Patching)

        # eq. 6 + 7:  S^r = Scorer^r(P~^s_max),  I^r = Argsort(S^r)
        shuffle_scores = self.scorer_shuffle(selected_patches)                  # [B, N, n, 1]   -- one score per selected patch
        shuffle_indices = torch.argsort(shuffle_scores, dim=-2, descending=True)# [B, N, n, 1]   -- sort indices by score (high → low)

        # eq. 8:  S^r_sort = S^r[I^r],  S^r_inv = detach(1/S^r_sort)
        shuffled_scores = torch.gather(shuffle_scores, dim=-2, index=shuffle_indices)   # [B, N, n, 1] -- pull out the sorted scores
        non_zero_mask = shuffled_scores != 0                                            # [B, N, n, 1] -- True where score != 0
        inv = (1 / shuffled_scores[non_zero_mask]).detach()                             # 1D flat     -- detached reciprocal (same trick as _select)

        # eq. 9 part 1:  P^r_sort = P~^s_max[I^r]
        shuffle_patch_indices = shuffle_indices.repeat(1, 1, 1, self.patch_len)         # [B, N, n, p] -- expand indices along patch_len
        shuffled_patches = torch.gather(selected_patches, dim=-2, index=shuffle_patch_indices)  # [B, N, n, p] -- reorder the patches

        # eq. 9 part 2 + eq. 10:  E^r = S^r_sort ⊙ S^r_inv,  P~ = P^r_sort ⊙ E^r
        # Same straight-through trick as _select: value=1, gradient flows through shuffled_scores.
        shuffled_scores[non_zero_mask] *= inv                                           # in-place    -- build E^r (value = 1)
        shuffled_patches = shuffled_scores * shuffled_patches                           # [B, N, n, p] -- multiply by "1" → bridges the gradient

        return shuffled_patches

    def forward(self, x):
        # x: [B, N, T]  -- channel-independent, already RevIN-normalized upstream by SRSNetModel
        n_vars = x.shape[1]                                             # scalar           -- number of channels (N)

        # Sec. 3.1 padding:  X -> X' in R^(N x (p + (n-1)·s))
        x = self.padding_patch_layer(x)                                 # [B, N, T_padded] -- pad on the right so the last patch fits

        # Reconstructive view P~ (Sec. 3.2 + 3.3 inside _rec_view)
        rec_repr_space = self._rec_view(x)                              # [B*N, n, p]      -- _select -> _shuffle

        # Original view P (conventional adjacent patching)
        original_repr_space = self._origin_view(x)                      # [B*N, n, p]      -- adjacent unfold with stride

        # eq. 13:  E~ = alpha ⊙ E^c + (1 - alpha) ⊙ E^s
        weight = torch.sigmoid(self.alpha)                              # [n, d_model]     -- alpha -> (0,1)^(n x d_model)
        embedding = weight * self.value_embedding_org(original_repr_space) \
                    + (1 - weight) * self.value_embedding_rec(rec_repr_space)   # [B*N, n, d_model] -- convex combo of the two embedded views

        # eq. 14:  E = E~ + PositionEmbedding(P)
        if self.pos:
            position_embedding = self.position_embedding(original_repr_space)   # [1, n, d_model]   -- sinusoidal PE (Vaswani 2017)
            embedding = embedding + position_embedding                          # [B*N, n, d_model] -- add position info

        # Dropout (not in the paper's formula, standard regularization)
        return self.dropout(embedding), n_vars                                  # [B*N, n, d_model], int
