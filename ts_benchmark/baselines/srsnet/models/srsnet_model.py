import math

import torch
from torch import nn

from ts_benchmark.baselines.srsnet.layers.SRS import SRS
from ts_benchmark.baselines.srsnet.layers.RevIN import RevIN


class FlattenHead(nn.Module):
    """Forecasting head: flatten the (d_model, patch_num) block and project to the horizon."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0, mode='linear'):
        """
        :param n_vars:        number of channels N
        :param nf:            d_model * patch_num  (size after flatten)
        :param target_window: horizon L (number of future steps to predict)
        :param head_dropout:  dropout on the head output
        :param mode:          'linear' = single Linear, anything else = 2-layer MLP
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)                                                     # collapse last two dims (d_model, patch_num)
        if mode == 'linear':
            self.head = nn.Linear(nf, target_window)                                                # one-shot regression  nf -> L
        else:
            # 2-layer MLP variant: nf -> nf/2 -> L  with SiLU activation
            self.head = nn.Sequential(nn.Linear(nf, nf // 2), nn.SiLU(), nn.Linear(nf // 2, target_window))
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, n_vars, d_model, patch_num]
        x = self.flatten(x)                                                                         # [B, n_vars, d_model * patch_num]
        x = self.head(x)                                                                            # [B, n_vars, target_window]
        x = self.dropout(x)
        return x


class SRSNetModel(nn.Module):
    """
    The actual nn.Module of SRSNet:  RevIN  ->  SRS  ->  FlattenHead  ->  RevIN^{-1}.

    Wrapped by the TFB-facing SRSNet class (srsnet.py). The SRS block is the
    paper's Selective Representation Space; RevIN is the Instance Normalization
    that Sec. 3.1 of the paper mentions ('first processed through Instance
    Normalization to mitigate the statistical differences between training and
    testing parts').
    """

    def __init__(self, config):
        """
        config exposes the paper-faithful hyperparameters:
            seq_len, pred_len, patch_len, stride, d_model, dropout, hidden_size,
            alpha, pos, enc_in, affine, subtract_last, head_mode
        """
        super(SRSNetModel, self).__init__()
        self.seq_len = config.seq_len                                                               # T: lookback window
        self.pred_len = config.pred_len                                                             # L: forecast horizon
        self.patch_len = config.patch_len                                                           # p: patch length
        self.stride = config.stride                                                                 # s: patch stride

        # --- The SRS block (Selective + Reassembly + Adaptive Fusion) --------
        self.patch_embedding = SRS(
            config.d_model, self.patch_len, self.stride, self.seq_len,
            config.dropout, config.hidden_size, config.alpha, config.pos
        )

        # --- Forecasting head ------------------------------------------------
        # After SRS we have [B, N, patch_num, d_model] -> flatten (d_model * patch_num) -> project to L
        self.head_nf = config.d_model * (math.ceil((config.seq_len - self.patch_len) / self.stride) + 1)  # nf = d_model * patch_num
        self.head = FlattenHead(
            config.enc_in,
            self.head_nf,
            config.pred_len,
            head_dropout=config.dropout,
            mode=config.head_mode
        )

        # --- Instance Normalization (RevIN, paper Sec. 3.1) -----------------
        self.revin = RevIN(num_features=config.enc_in, affine=config.affine, subtract_last=config.subtract_last)

    def forward(self, x_enc):
        # x_enc: [B, T, N]  -- this is the convention TFB uses for input

        # --- 1) Normalize per-instance, per-channel  (paper Sec. 3.1) -------
        x_enc = self.revin(x_enc, 'norm')                                                           # [B, T, N], mean=0 / std=1 per channel

        # --- 2) Move channel axis next to batch  (SRS expects [B, N, T]) ---
        x_enc = x_enc.permute(0, 2, 1)                                                              # [B, N, T]

        # --- 3) Apply the SRS block (encoder) -------------------------------
        enc_out, n_vars = self.patch_embedding(x_enc)                                               # [B*N, patch_num, d_model]

        # --- 4) Split batch and channel back apart --------------------------
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )                                                                                           # [B, N, patch_num, d_model]
        # FlattenHead wants the d_model axis next to patch_num
        enc_out = enc_out.permute(0, 1, 3, 2)                                                       # [B, N, d_model, patch_num]

        # --- 5) Forecasting head -------------------------------------------
        dec_out = self.head(enc_out)                                                                # [B, N, L]
        dec_out = dec_out.permute(0, 2, 1)                                                          # [B, L, N]  (back to TFB convention)

        # --- 6) De-normalize so the output is in the original scale --------
        dec_out = self.revin(dec_out, 'denorm')                                                     # [B, L, N]
        return dec_out
