import math

import torch
from torch import nn

from ts_benchmark.baselines.srsnet.layers.SRS import SRS
from ts_benchmark.baselines.srsnet.layers.RevIN import RevIN


class FlattenHead(nn.Module):
    """Forecasting head (paper Sec. 4, eq. 15-16): flatten the (d, n) block and project to L.

    Notation used throughout
    ========================

    Shape symbols
        B            batch size
        N            number of channels                       (= n_vars)
        n            number of patches                        (= patch_num)
        d            embedding dimension                      (= d_model)
        L            forecast horizon                         (= target_window)
        nf           = d * n                                  (flat size after collapse)

    Tensors
        x            input embedding block                    [B, N, d, n]   (from SRS)
        Flatten(x)   d and n collapsed into a single axis     [B, N, n*d]
        MLP(...)     one-shot linear projection to horizon    [B, N, L]

    Learned modules
        Flatten      nn.Flatten(start_dim=-2)                 (eq. 15 part 1)
        MLP          Linear(nf -> L) or 2-layer SiLU MLP      (eq. 15 part 2)
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0, mode='linear'):
        """
        n_vars:        number of channels N
        nf:            d * n  (size after Flatten)
        target_window: forecast horizon L
        head_dropout:  dropout probability applied on the head output
        mode:          'linear'  -> single Linear(nf -> L)
                       anything  -> 2-layer MLP nf -> nf/2 -> L  with SiLU
        """
        super().__init__()
        self.n_vars = n_vars

        # eq. 15:  Flatten : R^(N x n x d) -> R^(N x (n*d))
        self.flatten = nn.Flatten(start_dim=-2)                                                     # collapse the last two axes

        # eq. 15:  MLP : R^(N x (n*d)) -> R^(N x L)
        if mode == 'linear':
            self.head = nn.Linear(nf, target_window)                                                # one-shot linear projection
        else:
            # 2-layer SiLU variant: nf -> nf/2 -> L
            self.head = nn.Sequential(nn.Linear(nf, nf // 2), nn.SiLU(), nn.Linear(nf // 2, target_window))

        self.dropout = nn.Dropout(head_dropout)                                                     # float prob, no shape change

    def forward(self, x):
        # x  [B, N, d, n]   from SRS (after the reshape + permute in SRSNetModel)

        # eq. 16 (Flatten part):  collapse (d, n) -> d*n
        x = self.flatten(x)                                                                         # [B, N, n*d]

        # eq. 16 (MLP part):  Y_norm = MLP(Flatten(E))   (one-shot horizon)
        x = self.head(x)                                                                            # [B, N, L]

        x = self.dropout(x)                                                                         # [B, N, L]   regularization
        return x


class SRSNetModel(nn.Module):
    """The actual nn.Module of SRSNet:  RevIN -> SRS -> FlattenHead -> RevIN^{-1}.

    Wrapped by the TFB-facing SRSNet class (srsnet.py). The SRS block is the
    paper's Selective Representation Space; RevIN is the Instance Normalization
    that Sec. 3.1 of the paper mentions ("first processed through Instance
    Normalization to mitigate the statistical differences between training
    and testing parts"); FlattenHead is the simple linear/MLP head from Sec. 4.

    Notation used throughout
    ========================

    Shape symbols (sizes/scalars)
        B          batch size
        N          number of channels        (= config.enc_in)
        T          lookback length           (= config.seq_len)
        L          forecast horizon          (= config.pred_len)
        p          patch length              (= config.patch_len)
        s          stride                    (= config.stride)
        n          number of patches         = ceil((T - p) / s) + 1
        d          embedding dimension       (= config.d_model)
        nf         = d * n                   (Flatten output size)

    Tensor flow inside forward(x_enc)
        x_enc    [B, T, N]   TFB input convention
                 -> RevIN('norm')         [B, T, N]   centered, unit-std per (sample, channel)
                 -> permute(0, 2, 1)      [B, N, T]   channel-first for SRS
                 -> SRS                   [B*N, n, d] n patch embeddings per (sample, channel)
                 -> reshape + permute     [B, N, d, n]
                 -> FlattenHead           [B, N, L]   forecast still in normalized scale
                 -> permute(0, 2, 1)      [B, L, N]
                 -> RevIN('denorm')       [B, L, N]   back to original scale (returned)

    Sub-modules
        self.patch_embedding   SRS module     (paper Sec. 3.2-3.4)
        self.head              FlattenHead    (paper Sec. 4, eq. 15-16)
        self.revin             RevIN          (paper Sec. 3.1)
    """

    def __init__(self, config):
        """Build SRS + FlattenHead + RevIN from the merged TFB config.

        config exposes the paper-faithful hyper-parameters:
            seq_len, pred_len, patch_len, stride, d_model, dropout,
            hidden_size, alpha, pos, enc_in, affine, subtract_last, head_mode
        """
        super(SRSNetModel, self).__init__()
        self.seq_len = config.seq_len                                                               # T
        self.pred_len = config.pred_len                                                             # L
        self.patch_len = config.patch_len                                                           # p
        self.stride = config.stride                                                                 # s

        # SRS module (Selective Patching + Dynamic Reassembly + Adaptive Fusion).
        self.patch_embedding = SRS(
            config.d_model, self.patch_len, self.stride, self.seq_len,
            config.dropout, config.hidden_size, config.alpha, config.pos
        )

        # Forecasting head (eq. 15-16): nf = d * n -> L.
        self.head_nf = config.d_model * (math.ceil((config.seq_len - self.patch_len) / self.stride) + 1)
        self.head = FlattenHead(
            config.enc_in,
            self.head_nf,
            config.pred_len,
            head_dropout=config.dropout,
            mode=config.head_mode,
        )

        # Instance Normalization (RevIN, paper Sec. 3.1).
        self.revin = RevIN(num_features=config.enc_in, affine=config.affine, subtract_last=config.subtract_last)

    def forward(self, x_enc):
        # x_enc  [B, T, N]   TFB convention

        # Sec. 3.1:  RevIN normalization (mean=0, std=1 per sample-channel).
        x_enc = self.revin(x_enc, 'norm')                                                           # [B, T, N]

        # Move the channel axis next to batch (SRS expects channel-first).
        x_enc = x_enc.permute(0, 2, 1)                                                              # [B, N, T]

        # Sec. 3.2-3.4:  SRS produces n patch embeddings per (sample, channel).
        enc_out, n_vars = self.patch_embedding(x_enc)                                               # [B*N, n, d]

        # Split B*N back into B and N for the head.
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))        # [B, N, n, d]
        # FlattenHead expects d before n.
        enc_out = enc_out.permute(0, 1, 3, 2)                                                       # [B, N, d, n]

        # Sec. 4 eq. 15-16:  Y_norm = MLP(Flatten(E)).
        dec_out = self.head(enc_out)                                                                # [B, N, L]
        dec_out = dec_out.permute(0, 2, 1)                                                          # [B, L, N]   back to TFB convention

        # Sec. 3.1:  RevIN denormalization (back to original scale).
        dec_out = self.revin(dec_out, 'denorm')                                                     # [B, L, N]
        return dec_out
