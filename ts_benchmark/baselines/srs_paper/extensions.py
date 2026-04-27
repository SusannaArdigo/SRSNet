from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from ts_benchmark.baselines.srsnet.layers.SRS import SRS
from ts_benchmark.baselines.srsnet.models.srsnet_model import SRSNetModel
from ts_benchmark.baselines.srsnet.srsnet import MODEL_HYPER_PARAMS, SRSNet

class SRSRandomSP(SRS):

    def _select(self, x_rec):
        batch, n_vars, candidate_num, patch_size = x_rec.shape

        idx = torch.randint(low=0, high=candidate_num,
                            size=(batch, n_vars, 1, self.patch_num),
                            device=x_rec.device)

        gather_idx = idx.repeat(1, 1, patch_size, 1).permute(0, 1, 3, 2)

        return torch.gather(x_rec, dim=-2, index=gather_idx)

class SRSRandomSPNoShuffle(SRSRandomSP):

    def _shuffle(self, selected_patches):
        return selected_patches

class SRSRandomSPRandomShuffle(SRSRandomSP):

    def _shuffle(self, selected_patches):
        batch, n_vars, patch_num, patch_size = selected_patches.shape

        scores = torch.rand(batch, n_vars, patch_num, 1,
                            device=selected_patches.device)

        order = torch.argsort(scores, dim=-2, descending=True)

        gather_idx = order.repeat(1, 1, 1, patch_size)

        return torch.gather(selected_patches, dim=-2, index=gather_idx)

class SRSTimeAware(SRS):

    N_FEATURES = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = max(self.N_FEATURES * 4, 16)
        self.scorer_select = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.patch_num),
        )

    @staticmethod
    def _window_features(x_rec):
        spec = torch.fft.rfft(x_rec, dim=-1).abs()
        dominant = spec[..., 1:].max(dim=-1).values if spec.shape[-1] > 1 else spec[..., 0]

        mean = x_rec.mean(dim=-1, keepdim=True)
        x_c = x_rec - mean

        var = x_rec.var(dim=-1, unbiased=False)

        denom = x_c.pow(2).sum(dim=-1) + 1e-8
        ac1 = (x_c[..., 1:] * x_c[..., :-1]).sum(dim=-1) / denom

        t = torch.arange(x_rec.shape[-1], device=x_rec.device, dtype=x_rec.dtype)
        t_c = t - t.mean()
        slope_denom = (t_c * t_c).sum() + 1e-8
        slope = ((x_rec - mean) * t_c).sum(dim=-1) / slope_denom

        return torch.stack([dominant, ac1, var, slope], dim=-1)

    def _select(self, x_rec):

        feats = self._window_features(x_rec)

        scores  = self.scorer_select(feats)
        indices = torch.argmax(scores, dim=-2, keepdim=True)

        max_scores    = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv           = (1 / max_scores[non_zero_mask]).detach()

        x_rec_indices    = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)

        max_scores[non_zero_mask] *= inv

        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches

        return selected_patches

class SRSHypernetAF(SRS):

    HYPER_HIDDEN = 8
    INIT_ALPHA = 3.5

    def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                 hidden_size, alpha=2.0, pos=True):
        super().__init__(d_model, patch_len, stride, seq_len, dropout,
                         hidden_size, alpha, pos)

        if hasattr(self, "alpha"):
            self.register_buffer("_unused_alpha", torch.zeros_like(self.alpha.data))
            del self.alpha

        self.context_proj = nn.Linear(d_model, self.HYPER_HIDDEN)
        self.hyper        = nn.Linear(self.HYPER_HIDDEN, self.patch_num * d_model)

        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.zeros_(self.hyper.weight)
        nn.init.constant_(self.hyper.bias, self.INIT_ALPHA)

    def forward(self, x):
        n_vars = x.shape[1]

        x = self.padding_patch_layer(x)

        rec_repr_space      = self._rec_view(x)
        original_repr_space = self._origin_view(x)

        e_orig = self.value_embedding_org(original_repr_space)
        e_rec  = self.value_embedding_rec(rec_repr_space)

        ctx       = e_orig.mean(dim=0).mean(dim=0)
        h         = torch.relu(self.context_proj(ctx))
        alpha_dyn = self.hyper(h).view(self.patch_num, e_orig.shape[-1])
        weight    = torch.sigmoid(alpha_dyn)

        embedding = weight * e_orig + (1.0 - weight) * e_rec

        if self.pos:
            embedding = embedding + self.position_embedding(original_repr_space)

        return self.dropout(embedding), n_vars

def _make_hypernet_combo(base_select_cls, combo_name):

    class _Combo(base_select_cls):

        HYPER_HIDDEN = SRSHypernetAF.HYPER_HIDDEN
        INIT_ALPHA = SRSHypernetAF.INIT_ALPHA

        def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                     hidden_size, alpha=2.0, pos=True, **extra_kwargs):
            super().__init__(d_model, patch_len, stride, seq_len, dropout,
                             hidden_size, alpha, pos, **extra_kwargs)
            if hasattr(self, "alpha"):
                self.register_buffer("_unused_alpha", torch.zeros_like(self.alpha.data))
                del self.alpha
            self.context_proj = nn.Linear(d_model, self.HYPER_HIDDEN)
            self.hyper = nn.Linear(self.HYPER_HIDDEN, self.patch_num * d_model)
            nn.init.zeros_(self.context_proj.weight)
            nn.init.zeros_(self.context_proj.bias)
            nn.init.zeros_(self.hyper.weight)
            nn.init.constant_(self.hyper.bias, self.INIT_ALPHA)

        def forward(self, x):
            n_vars = x.shape[1]
            x = self.padding_patch_layer(x)
            rec_repr_space = self._rec_view(x)
            original_repr_space = self._origin_view(x)
            e_orig = self.value_embedding_org(original_repr_space)
            e_rec = self.value_embedding_rec(rec_repr_space)
            ctx = e_orig.mean(dim=0).mean(dim=0)
            h = torch.relu(self.context_proj(ctx))
            alpha_dyn = self.hyper(h).view(self.patch_num, e_orig.shape[-1])
            weight = torch.sigmoid(alpha_dyn)
            embedding = weight * e_orig + (1.0 - weight) * e_rec
            if self.pos:
                embedding = embedding + self.position_embedding(original_repr_space)
            return self.dropout(embedding), n_vars

    _Combo.__name__ = combo_name
    _Combo.__qualname__ = combo_name
    return _Combo

SRSTimeAware_HypernetAF = _make_hypernet_combo(
    SRSTimeAware, "SRSTimeAware_HypernetAF"
)
SRSRandomSP_HypernetAF = _make_hypernet_combo(
    SRSRandomSP, "SRSRandomSP_HypernetAF"
)

def _make_3way_combo(two_way_combo_cls, combo_name):

    class _ThreeWayCombo(two_way_combo_cls):

        def _shuffle(self, selected_patches):
            return selected_patches

    _ThreeWayCombo.__name__ = combo_name
    _ThreeWayCombo.__qualname__ = combo_name
    return _ThreeWayCombo

SRSTimeAware_NoShuffle_HypernetAF = _make_3way_combo(
    SRSTimeAware_HypernetAF, "SRSTimeAware_NoShuffle_HypernetAF"
)
SRSRandomSP_NoShuffle_HypernetAF = _make_3way_combo(
    SRSRandomSP_HypernetAF, "SRSRandomSP_NoShuffle_HypernetAF"
)

class _SelectivityControlsSRSNetModel(SRSNetModel):

    embedding_cls = None

    def __init__(self, config):
        super().__init__(config)
        if self.embedding_cls is None:
            raise NotImplementedError("Subclass must set embedding_cls to an SRS variant.")
        kwargs = dict(
            d_model=config.d_model, patch_len=self.patch_len, stride=self.stride,
            seq_len=self.seq_len, dropout=config.dropout, hidden_size=config.hidden_size,
            alpha=config.alpha, pos=config.pos,
        )
        self.patch_embedding = self.embedding_cls(**kwargs)

class SRSNet_RandomSP_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSP

class SRSNet_RandomSPNoShuffle_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSPNoShuffle

class SRSNet_RandomSPRandomShuffle_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSPRandomShuffle

class SRSNet_TASP_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSTimeAware

class SRSNet_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSHypernetAF

class SRSNet_TASP_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSTimeAware_HypernetAF

class SRSNet_RandomSP_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSP_HypernetAF

class SRSNet_TASP_NoShuffle_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSTimeAware_NoShuffle_HypernetAF

class SRSNet_RandomSP_NoShuffle_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSP_NoShuffle_HypernetAF

class _SelectivityControlsSRSNet(SRSNet):

    variant_name = None
    model_cls = None

    @property
    def model_name(self):
        return self.variant_name

    def _init_model(self):
        return self.model_cls(self.config)

class SRSNet_RandomSP(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_RandomSP"
    model_cls = SRSNet_RandomSP_Model

class SRSNet_RandomSPNoShuffle(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_RandomSPNoShuffle"
    model_cls = SRSNet_RandomSPNoShuffle_Model

class SRSNet_RandomSPRandomShuffle(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_RandomSPRandomShuffle"
    model_cls = SRSNet_RandomSPRandomShuffle_Model

class SRSNet_TASP(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_TASP"
    model_cls = SRSNet_TASP_Model

class SRSNet_HypernetAF(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_HypernetAF"
    model_cls = SRSNet_HypernetAF_Model

class SRSNet_TASP_HypernetAF(_SelectivityControlsSRSNet):

    variant_name = "SRSNet_TASP_HypernetAF"
    model_cls = SRSNet_TASP_HypernetAF_Model

class SRSNet_RandomSP_HypernetAF(_SelectivityControlsSRSNet):

    variant_name = "SRSNet_RandomSP_HypernetAF"
    model_cls = SRSNet_RandomSP_HypernetAF_Model

class SRSNet_TASP_NoShuffle_HypernetAF(_SelectivityControlsSRSNet):

    variant_name = "SRSNet_TASP_NoShuffle_HypernetAF"
    model_cls = SRSNet_TASP_NoShuffle_HypernetAF_Model

class SRSNet_RandomSP_NoShuffle_HypernetAF(_SelectivityControlsSRSNet):

    variant_name = "SRSNet_RandomSP_NoShuffle_HypernetAF"
    model_cls = SRSNet_RandomSP_NoShuffle_HypernetAF_Model

class _SRSNetWithEncoderModel(SRSNetModel):

    def __init__(self, config):
        super().__init__(config)
        n_heads  = getattr(config, "encoder_n_heads", 8)
        n_layers = getattr(config, "encoder_n_layers", 2)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x_enc):

        x_enc = self.revin(x_enc, 'norm')

        x_enc = x_enc.permute(0, 2, 1)

        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out).permute(0, 2, 1)

        return self.revin(dec_out, 'denorm')

class _SelectivityControlsSRSNetEncoderModel(_SelectivityControlsSRSNetModel):

    def __init__(self, config):
        super().__init__(config)
        n_heads  = getattr(config, "encoder_n_heads", 8)
        n_layers = getattr(config, "encoder_n_layers", 2)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x_enc):
        x_enc = self.revin(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1)

        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out = self.encoder(enc_out)

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        return self.revin(dec_out, 'denorm')

class SRSNet_TASP_TransformerEncoder_Model(_SelectivityControlsSRSNetEncoderModel):
    embedding_cls = SRSTimeAware

class SRSNet_HypernetAF_TransformerEncoder_Model(_SelectivityControlsSRSNetEncoderModel):
    embedding_cls = SRSHypernetAF

class SRSNet_TransformerEncoder(_SelectivityControlsSRSNet):

    variant_name = "SRSNet_TransformerEncoder"
    model_cls = _SRSNetWithEncoderModel

class SRSNet_TASP_TransformerEncoder(_SelectivityControlsSRSNet):

    variant_name = "SRSNet_TASP_TransformerEncoder"
    model_cls = SRSNet_TASP_TransformerEncoder_Model

class SRSNet_HypernetAF_TransformerEncoder(_SelectivityControlsSRSNet):

    variant_name = "SRSNet_HypernetAF_TransformerEncoder"
    model_cls = SRSNet_HypernetAF_TransformerEncoder_Model

SRSNet_RandomSRS = SRSNet_RandomSP

for _cls in (
    SRSNet_RandomSP,
    SRSNet_RandomSPNoShuffle,
    SRSNet_RandomSPRandomShuffle,
    SRSNet_TASP,
    SRSNet_HypernetAF,
    SRSNet_TASP_HypernetAF,
    SRSNet_RandomSP_HypernetAF,
    SRSNet_TASP_NoShuffle_HypernetAF,
    SRSNet_RandomSP_NoShuffle_HypernetAF,
    SRSNet_TransformerEncoder,
    SRSNet_TASP_TransformerEncoder,
    SRSNet_HypernetAF_TransformerEncoder,
):
    _cls.MODEL_HYPER_PARAMS = MODEL_HYPER_PARAMS

__all__ = [
    "SRSRandomSP",
    "SRSRandomSPNoShuffle",
    "SRSRandomSPRandomShuffle",
    "SRSTimeAware",
    "SRSHypernetAF",
    "SRSTimeAware_HypernetAF",
    "SRSRandomSP_HypernetAF",
    "SRSTimeAware_NoShuffle_HypernetAF",
    "SRSRandomSP_NoShuffle_HypernetAF",
    "SRSNet_RandomSP",
    "SRSNet_RandomSPNoShuffle",
    "SRSNet_RandomSPRandomShuffle",
    "SRSNet_TASP",
    "SRSNet_HypernetAF",
    "SRSNet_TASP_HypernetAF",
    "SRSNet_RandomSP_HypernetAF",
    "SRSNet_TASP_NoShuffle_HypernetAF",
    "SRSNet_RandomSP_NoShuffle_HypernetAF",
    "SRSNet_TransformerEncoder",
    "SRSNet_TASP_TransformerEncoder",
    "SRSNet_HypernetAF_TransformerEncoder",
    "SRSNet_RandomSRS",
]
