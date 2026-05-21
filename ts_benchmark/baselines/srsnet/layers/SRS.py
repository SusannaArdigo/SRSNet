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
    """Sinusoidal positional embedding (Vaswani et al. 2017).

    Precomputes a [1, max_len, d_model] table at init; forward slices
    the first x.size(1) rows. Read-only buffer (no gradient).
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()                          # pe   [max_len, d_model]
        pe.require_grad = False

        # Standard Vaswani formula:
        #   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        #   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        position = torch.arange(0, max_len).float().unsqueeze(1)            # [max_len, 1]
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()                 # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)                        # even dims -> sin
        pe[:, 1::2] = torch.cos(position * div_term)                        # odd dims  -> cos

        pe = pe.unsqueeze(0)                                                # [1, max_len, d_model]
        self.register_buffer('pe', pe)                                      # non-trainable buffer

    def forward(self, x):
        # Slice the first n positions, ignoring x's actual content.
        return self.pe[:, :x.size(1)]                                       # [1, n, d_model]


class SRS(nn.Module):
    """Selective Representation Space block (Wu et al., NeurIPS 2025).

    Pipeline per batch (input arrives already RevIN-normalized by SRSNetModel):
        1. Pad the input on the right so the last patch fits exactly.       (Sec. 3.1)
        2. Original view P     -- adjacent patching with stride             (Sec. 3.1)
        3. Reconstructive view P~  -- stride-1 candidates + select + shuffle (Sec. 3.2 eq. 1-5  +  Sec. 3.3 eq. 6-10)
        4. Embed both views with two separate Linear projections             (Sec. 3.4 eq. 11-12)
        5. Adaptive Fusion: learned convex combination via sigmoid(alpha)    (Sec. 3.4 eq. 13)
        6. Add positional embedding, dropout, return                         (Sec. 3.4 eq. 14)

    Paper equations are referenced inline (e.g. `# eq. 2:`) right above each
    implementation line.

    Notation used throughout
    ========================

    Shape symbols (sizes/scalars)
        B          batch size
        N          number of channels
        T          lookback length (input window length)
        T_pad      = T + padding             (after right-padding)
        padding    = p + (n - 1) * s - T     (samples appended on the right)
        L          forecast horizon
        p          patch length              (= patch_len)
        s          stride between adjacent patches
        n          number of adjacent patches  = ceil((T - p) / s) + 1   (= patch_num)
        K          number of stride=1 candidate patches  = (n - 1) * s + 1
        d          embedding dimension       (= d_model)
        h          scorer hidden size
        max_len    upper bound used to precompute sinusoidal PE (5000 by default)

    Input / intermediate / output tensors
        X          input lookback             [B, N, T]            (RevIN-normalized upstream)
        X'         right-padded X             [B, N, T_pad]
        P          original view: adjacent patches with stride s   [B*N, n, p]
        P'         all K stride=1 candidate patches                [B, N, K, p]
        P~^s_max   the n selected candidates (output of _select)   [B, N, n, p]
        P~         P~^s_max reordered by _shuffle                  [B*N, n, p]
        E^c        PatchEmbedding^1(P)                             [B*N, n, d]
        E^s_emb    PatchEmbedding^2(P~)                            [B*N, n, d]
        E~         alpha * E^c + (1 - alpha) * E^s_emb              [B*N, n, d]
        E          E~ + sinusoidal PositionEmbedding(P)            [B*N, n, d]   (returned)

    Learned modules (paper symbol -> code attribute)
        Scorer^s          self.scorer_select         MLP: p -> hidden -> n   (eq. 1)
        Scorer^r          self.scorer_shuffle        MLP: p -> hidden -> 1   (eq. 6)
        PatchEmbedding^1  self.value_embedding_org   Linear: p -> d          (eq. 11)
        PatchEmbedding^2  self.value_embedding_rec   Linear: p -> d          (eq. 11)
        PositionEmbedding self.position_embedding    sinusoidal (Vaswani)    (eq. 14)
        alpha             self.alpha (Parameter)     learned fusion weight   [n, d]   (eq. 13)

    Selective Patching intermediates (eq. 1-5, used inside _select)
        S^s        Scorer^s(P')                                    [B, N, K, n]
        I^s        Argmax_K(S^s)                                   [B, N, 1, n]
        S^s_max    S^s[I^s]                                        [B, N, 1, n]
        S^s_inv    detach(1 / S^s_max)                             flat (no grad)
        P^s_max    P'[I^s]                                         [B, N, n, p]
        E^s        S^s_max * S^s_inv   (= 1 numerically, grad alive; STE bridge)

    Dynamic Reassembly intermediates (eq. 6-10, used inside _shuffle)
        S^r        Scorer^r(P~^s_max)                              [B, N, n, 1]
        I^r        Argsort_n(S^r)                                  [B, N, n, 1]
        S^r_sort   S^r[I^r]                                        [B, N, n, 1]
        S^r_inv    detach(1 / S^r_sort)                            flat (no grad)
        P^r_sort   P~^s_max[I^r]                                   [B, N, n, p]
        E^r        S^r_sort * S^r_inv   (STE bridge for reorder)
    """

    def __init__(self, d_model, patch_len, stride, seq_len, dropout, hidden_size, alpha=2.0, pos=True):
        """Build all SRS components.

        Paper notation -> code attribute:
            p, s, T              patch_len, stride, seq_len            input geometry
            n                    patch_num = ceil((T - p)/s) + 1       #adjacent patches
            X -> X'              padding_patch_layer                    right-pad to (n-1)*s + p
            Scorer^s             scorer_select                          MLP for Selective Patching (eq. 1)
            Scorer^r             scorer_shuffle                         MLP for Dynamic Reassembly (eq. 6)
            PatchEmbedding^1     value_embedding_org                    embed P (eq. 11+12)
            PatchEmbedding^2     value_embedding_rec                    embed P~ (eq. 11+12)
            PositionEmbedding    position_embedding                     sinusoidal PE (eq. 14)
            alpha                self.alpha (learned)                   fusion weight (eq. 13)
        """
        super(SRS, self).__init__()

        # Patching geometry: from p, s, T derive n and the right-padding amount.
        self.patch_len = patch_len                                                 # p
        self.stride = stride                                                       # s
        self.seq_len = seq_len                                                     # T
        # n = number of adjacent patches once we right-pad X to fit cleanly.
        self.patch_num = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1
        # padding = how many samples to append so the last patch fits exactly.
        self.padding = self.patch_len + (self.patch_num - 1) * self.stride - self.seq_len
        # ReplicationPad1d repeats the last value (paper-compatible).
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))           # X -> X'  [B, N, T] -> [B, N, T_pad]

        # Scorer^s (Selective Patching, eq. 1): for each candidate produce n scores,
        # so argmax over K picks the winner per output slot.
        self.scorer_select = nn.Sequential(                                         # Scorer^s   [B, N, K, p] -> [B, N, K, n]
            nn.Linear(self.patch_len, hidden_size), nn.ReLU(),                      # p -> h
            nn.Linear(hidden_size, self.patch_num),                                 # h -> n
        )

        # Scorer^r (Dynamic Reassembly, eq. 6): one score per selected patch,
        # argsort over n gives the new order.
        self.scorer_shuffle = nn.Sequential(                                        # Scorer^r   [B, N, n, p] -> [B, N, n, 1]
            nn.Linear(self.patch_len, hidden_size), nn.ReLU(),                      # p -> h
            nn.Linear(hidden_size, 1),                                              # h -> 1
        )

        # PatchEmbedding^1 and PatchEmbedding^2 (eq. 11+12): two parallel projections,
        # one per view (original P / reconstructive P~), so Adaptive Fusion has two
        # embeddings to combine.
        self.value_embedding_org = nn.Linear(patch_len, d_model, bias=False)        # PatchEmbedding^1   p -> d
        self.value_embedding_rec = nn.Linear(patch_len, d_model, bias=False)        # PatchEmbedding^2   p -> d

        # Sinusoidal positional embedding (eq. 14).
        if pos:
            self.position_embedding = PositionalEmbedding(d_model)                  # PE buffer  [1, max_len, d]
        self.pos = pos

        # Final dropout on the fused embedding (not in paper, standard regularization).
        self.dropout = nn.Dropout(dropout)

        # alpha (Adaptive Fusion weight, eq. 13).
        # Init at 2.0 -> sigmoid(2.0) ~ 0.88: at step 0 the model trusts P (original view).
        self.alpha = nn.Parameter(torch.ones(self.patch_num, d_model) * alpha)      # alpha   [n, d]

    def _origin_view(self, x):
        """Conventional adjacent patching (paper Sec. 3.1): produces P.

        Paper notation:
            X'  the right-padded input
            P   the adjacent patches with stride s  (returned, flattened on batch*channel)
        """
        # X' = x  [B, N, T_padded]

        # Unfold along time with step=stride -> non-overlapping (or stride-s) patches
        x_origin = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # P (4D)   [B, N, n, p]

        # Flatten batch and channel (channel-independent: every channel handled the same way)
        return rearrange(x_origin, 'b c n p -> (b c) n p')                         # P        [B*N, n, p]

    def _rec_view(self, x):
        """Reconstructive view (paper Sec. 3.2 + 3.3): _select then _shuffle.

        Paper notation:
            X'         the right-padded input
            P'         all K stride=1 candidate patches  (intermediate)
            P~^s_max   the n selected patches  (from _select)
            P~         the n reordered patches  (returned, flattened on batch*channel)
        """
        # X' = x  [B, N, T_padded]

        # Sec. 3.2 setup:  P' = unfold(X', stride=1)  -- all K candidate patches
        x_rec = x.unfold(dimension=-1, size=self.patch_len, step=1)                # P'         [B, N, K, p]

        # Sec. 3.2:  P~^s_max = Selective Patching of P'  (eq. 1-5)
        selected_patches = self._select(x_rec)                                     # P~^s_max   [B, N, n, p]
        # Sec. 3.3:  P~ = Dynamic Reassembly of P~^s_max  (eq. 6-10)
        shuffled_patches = self._shuffle(selected_patches)                         # P~         [B, N, n, p]

        # Flatten batch and channel for consistency with _origin_view
        return rearrange(shuffled_patches, 'b c n p -> (b c) n p')                 # P~         [B*N, n, p]

    def _select(self, x_rec):
        """Selective Patching (paper Sec. 3.2): pick n best candidates out of K.

        Paper notation:
            P'        all K stride=1 candidate patches  (input)
            Scorer^s  MLP that scores patches
            S^s       full score tensor
            I^s       argmax indices: which candidate fills each slot
            S^s_max   score of the winning candidate per slot
            S^s_inv   detached 1/S^s_max (no gradient)
            P^s_max   the n winning patches
            E^s       "1 with live gradient" (straight-through bridge)
            P~^s_max  STE-bridged winning patches (returned)

        Shapes: B=batch, N=channels, K=#candidates, n=patch_num, p=patch_len.
        """
        # P' = x_rec  [B, N, K, p]

        # eq. 1+2:  S^s = Scorer^s(P'),   I^s = Argmax_K(S^s)
        scores  = self.scorer_select(x_rec)                                 # S^s   [B, N, K, n]
        indices = torch.argmax(scores, dim=-2, keepdim=True)                # I^s   [B, N, 1, n]

        # eq. 3:   S^s_max = S^s[I^s],   S^s_inv = detach(1/S^s_max)
        max_scores    = torch.gather(input=scores, dim=-2, index=indices)   # S^s_max   [B, N, 1, n]
        non_zero_mask = max_scores != 0                                     # skip rare zero scores
        inv           = (1 / max_scores[non_zero_mask]).detach()            # S^s_inv   flat   (.detach blocks grad)

        # eq. 4 (first half):  P^s_max = P'[I^s]
        x_rec_indices    = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)  # broadcast on p
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)       # P^s_max   [B, N, n, p]

        # eq. 4 (second half):  E^s = S^s_max * S^s_inv   (numerically = 1, grad alive)
        max_scores[non_zero_mask] *= inv                                                # E^s   [B, N, 1, n]

        # eq. 5:   P~^s_max = P^s_max * E^s   (straight-through: value unchanged, grad bridged)
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches            # P~^s_max  [B, N, n, p]

        return selected_patches

    def _shuffle(self, selected_patches):
        """Dynamic Reassembly (paper Sec. 3.3): reorder the n patches by learned scores.

        Paper notation:
            P~^s_max  the n selected patches  (input, from _select)
            Scorer^r  MLP that scores patches  (single score per patch)
            S^r       full score vector (one score per patch)
            I^r       argsort indices: the new patch order
            S^r_sort  sorted scores (high to low)
            S^r_inv   detached 1/S^r_sort (no gradient)
            P^r_sort  the n patches in the new order
            E^r       "1 with live gradient" (straight-through bridge)
            P~        STE-bridged reordered patches (returned)

        Shapes: B=batch, N=channels, n=patch_num, p=patch_len.
        """
        # P~^s_max = selected_patches  [B, N, n, p]

        # eq. 6+7:  S^r = Scorer^r(P~^s_max),   I^r = Argsort(S^r)  (descending)
        shuffle_scores  = self.scorer_shuffle(selected_patches)                          # S^r   [B, N, n, 1]
        shuffle_indices = torch.argsort(shuffle_scores, dim=-2, descending=True)         # I^r   [B, N, n, 1]

        # eq. 8:   S^r_sort = S^r[I^r],   S^r_inv = detach(1/S^r_sort)
        shuffled_scores = torch.gather(shuffle_scores, dim=-2, index=shuffle_indices)    # S^r_sort   [B, N, n, 1]
        non_zero_mask   = shuffled_scores != 0                                           # skip rare zero scores
        inv             = (1 / shuffled_scores[non_zero_mask]).detach()                  # S^r_inv   flat   (.detach blocks grad)

        # eq. 9 (first half):  P^r_sort = P~^s_max[I^r]
        shuffle_patch_indices = shuffle_indices.repeat(1, 1, 1, self.patch_len)          # broadcast on p
        shuffled_patches      = torch.gather(selected_patches, dim=-2,
                                             index=shuffle_patch_indices)                # P^r_sort   [B, N, n, p]

        # eq. 9 (second half):  E^r = S^r_sort * S^r_inv   (numerically = 1, grad alive)
        shuffled_scores[non_zero_mask] *= inv                                            # E^r   [B, N, n, 1]

        # eq. 10:  P~ = P^r_sort * E^r   (straight-through: value unchanged, grad bridged)
        shuffled_patches = shuffled_scores * shuffled_patches                            # P~    [B, N, n, p]

        return shuffled_patches

    def forward(self, x):
        """Full SRS forward: pad -> two views -> embed -> fuse -> position -> dropout.

        Paper notation:
            X         input lookback window  (RevIN-normalized upstream)
            X'        right-padded X
            P         original view  (adjacent patches)
            P~        reconstructive view  (selective + shuffled patches)
            E^c       PatchEmbedding^1(P)
            E^s       PatchEmbedding^2(P~)
            alpha     learned per-position, per-channel fusion weight in (0, 1)
            E~        convex combination of E^c and E^s
            E         E~ + sinusoidal positional embedding (returned)

        Shapes: B=batch, N=channels, T=lookback, T_pad=padded length,
                n=patch_num, p=patch_len, d=d_model.
        """
        # X = x  [B, N, T]
        n_vars = x.shape[1]                                             # N = number of channels

        # Sec. 3.1:  X' = pad_right(X)
        x = self.padding_patch_layer(x)                                 # X'   [B, N, T_pad]

        # Sec. 3.2+3.3:  P~ from _rec_view  (calls _select then _shuffle)
        rec_repr_space      = self._rec_view(x)                         # P~   [B*N, n, p]
        # Sec. 3.1:       P from _origin_view  (adjacent stride-s patches)
        original_repr_space = self._origin_view(x)                      # P    [B*N, n, p]

        # eq. 12+13:  E^c = PatchEmbedding^1(P),  E^s = PatchEmbedding^2(P~),
        #             E~ = alpha * E^c + (1 - alpha) * E^s
        weight = torch.sigmoid(self.alpha)                              # alpha   [n, d]   in (0, 1)
        embedding = weight * self.value_embedding_org(original_repr_space) \
                    + (1 - weight) * self.value_embedding_rec(rec_repr_space)   # E~   [B*N, n, d]

        # eq. 14:  E = E~ + PositionEmbedding(P)
        if self.pos:
            position_embedding = self.position_embedding(original_repr_space)   # PE   [1, n, d]
            embedding = embedding + position_embedding                          # E    [B*N, n, d]

        # Dropout (not in paper, standard regularization)
        return self.dropout(embedding), n_vars                                  # [B*N, n, d], int
