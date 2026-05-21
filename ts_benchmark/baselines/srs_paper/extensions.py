"""Our constructive extensions to SRSNet + their factorial combos.

Five families, all mapped to a specific paper Future-Work (FW) or
Limitation (L) item from Sec. 6 of the SRSNet paper:

  - Random controls (negative-control study)
        SRSRandomSP, SRSRandomSPNoShuffle, SRSRandomSPRandomShuffle
        WHY: sanity check that the LEARNED scorer matters at all.
  - TASP (Time-Aware Selective Patching)               -> FW#1 + L3
        WHY: replace the opaque scorer MLP with engineered per-window
        features (FFT, autocorr, variance, slope). Tests if a smaller,
        interpretable scorer matches or beats the learned one.
  - Hypernet-AF                                        -> FW#3 + L4
        WHY: replace the static [n, d] alpha with a per-instance value
        from a tiny hypernet over a batch-level context vector. Tests
        whether data-dependent fusion helps.
  - Factorial combos: (Select variant) x HypernetAF
        WHY: fill the missing cells of the (Select, Fusion) 3x2 table.
  - 3-axis combos: (Select variant) x identity Shuffle x HypernetAF
        WHY: isolate whether the learned _shuffle is doing anything
        that the 2-axis combos overlooked.
  - Backbone extension: SRSNet + TransformerEncoder + FlattenHead
        WHY: tests the paper's claim (Sec. 4) that a linear head is
        enough by inserting a real Transformer Encoder.

Backwards-compat: ``SRSNet_RandomSRS`` is aliased to ``SRSNet_RandomSP``
so old manifests still resolve. New code should use the precise name.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from ts_benchmark.baselines.srsnet.layers.SRS import SRS
from ts_benchmark.baselines.srsnet.models.srsnet_model import SRSNetModel
from ts_benchmark.baselines.srsnet.srsnet import MODEL_HYPER_PARAMS, SRSNet


# ---------------------------------------------------------------------------
# Random selective-patching layers
# ---------------------------------------------------------------------------
class SRSRandomSP(SRS):
    """Random ``_select``: uniformly random index for each output slot.

    WHY: negative-control. Tests if the learned scorer matters at all.
    The shuffle is kept learned so the only thing changing is selection.
    Scorer params still exist (shape compat) but get no gradient.
    """

    def _select(self, x_rec):
        # x_rec: [B, N, K, p] -- random index per output slot, no scores
        batch, n_vars, candidate_num, patch_size = x_rec.shape
        idx = torch.randint(low=0, high=candidate_num,
                            size=(batch, n_vars, 1, self.patch_num),
                            device=x_rec.device)
        gather_idx = idx.repeat(1, 1, patch_size, 1).permute(0, 1, 3, 2)   # broadcast on patch_len
        return torch.gather(x_rec, dim=-2, index=gather_idx)               # [B, N, n, p]


class SRSRandomSPNoShuffle(SRSRandomSP):
    """Random ``_select`` + identity ``_shuffle``.

    WHY: cleanest "no learning anywhere in SRS" control. No randomness
    in the shuffle stage, so all variance comes from _select.
    """

    def _shuffle(self, selected_patches):
        return selected_patches                                            # identity -- no reorder


class SRSRandomSPRandomShuffle(SRSRandomSP):
    """Random ``_select`` + random ``_shuffle``.

    WHY: full-chaos baseline. Higher per-seed variance than NoShuffle
    because both stages re-sample every forward.
    """

    def _shuffle(self, selected_patches):
        batch, n_vars, patch_num, patch_size = selected_patches.shape
        scores = torch.rand(batch, n_vars, patch_num, 1, device=selected_patches.device)
        order = torch.argsort(scores, dim=-2, descending=True)             # random permutation
        gather_idx = order.repeat(1, 1, 1, patch_size)
        return torch.gather(selected_patches, dim=-2, index=gather_idx)


# ---------------------------------------------------------------------------
# Constructive extensions: TASP, Hypernet-AF
# ---------------------------------------------------------------------------
class SRSTimeAware(SRS):
    """TASP: scorer over 4 engineered per-window features (FW#1 + L3).

    WHY: tests whether a tiny, *interpretable* scorer matches or beats
    the opaque learned-over-raw-values scorer of vanilla SRS. Inputs are
    dominant FFT magnitude, lag-1 autocorrelation, variance, trend slope.
    Scorer params drop from ~6k to ~370, so any gain is NOT from capacity.
    Shuffle scorer is untouched -- changes are attributable to _select only.
    """

    N_FEATURES = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap the [patch_len -> hidden -> patch_num] MLP for a
        # [4 -> 16 -> patch_num] MLP over engineered features.
        hidden = max(self.N_FEATURES * 4, 16)
        self.scorer_select = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.patch_num),
        )

    @staticmethod
    def _window_features(x_rec):
        """Compute (FFT max, lag-1 autocorr, var, trend slope) per window.

        x_rec: [B, C, candidate_num, patch_size] -> [B, C, candidate_num, 4]
        """
        # 1) Dominant non-DC FFT magnitude (seasonality strength)
        spec = torch.fft.rfft(x_rec, dim=-1).abs()
        dominant = spec[..., 1:].max(dim=-1).values if spec.shape[-1] > 1 else spec[..., 0]
        # 2) variance + 3) lag-1 autocorrelation (smoothness)
        mean = x_rec.mean(dim=-1, keepdim=True)
        var = x_rec.var(dim=-1, unbiased=False)
        x_c = x_rec - mean
        denom = x_c.pow(2).sum(dim=-1) + 1e-8
        ac1 = (x_c[..., 1:] * x_c[..., :-1]).sum(dim=-1) / denom
        # 4) Trend slope via least-squares (linear-fit gradient)
        t = torch.arange(x_rec.shape[-1], device=x_rec.device, dtype=x_rec.dtype)
        t_c = t - t.mean()
        slope_denom = (t_c * t_c).sum() + 1e-8
        slope = ((x_rec - mean) * t_c).sum(dim=-1) / slope_denom
        return torch.stack([dominant, ac1, var, slope], dim=-1)

    def _select(self, x_rec):
        """Same straight-through select as vanilla SRS, but scorer eats features not values."""
        feats = self._window_features(x_rec)                                    # [B, C, K, 4]
        scores = self.scorer_select(feats)                                      # [B, C, K, n]
        indices = torch.argmax(scores, dim=-2, keepdim=True)                    # I^s (eq. 2)
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()                          # detached reciprocal
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)
        max_scores[non_zero_mask] *= inv                                        # STE: value=1, grad alive
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches
        return selected_patches


class SRSHypernetAF(SRS):
    """Hypernet-AF: data-dependent alpha via a tiny hypernet (FW#3 + L4).

    WHY: replace the static [n, d] alpha with one computed per-instance.
    A tiny hypernet eats a batch-level context vector (mean-pooled embedding)
    and outputs the alpha. Init: zero weights + bias=3.5 so sigmoid(alpha)
    starts at ~0.97 (same as vanilla SRSNet at step 0) -- any gain comes
    from learned context-dependence, not from capacity at init.
    """

    HYPER_HIDDEN = 8
    INIT_ALPHA = 3.5

    def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                 hidden_size, alpha=2.0, pos=True):
        super().__init__(d_model, patch_len, stride, seq_len, dropout,
                         hidden_size, alpha, pos)
        # Replace the free alpha with the hypernet pattern. Keep a buffer
        # under a placeholder name so state-dict shape stays compatible.
        if hasattr(self, "alpha"):
            self.register_buffer("_unused_alpha", torch.zeros_like(self.alpha.data))
            del self.alpha
        self.context_proj = nn.Linear(d_model, self.HYPER_HIDDEN)              # context -> hidden
        self.hyper = nn.Linear(self.HYPER_HIDDEN, self.patch_num * d_model)    # hidden -> alpha (flat)
        # Vanilla-preserving init: zero weights + bias=INIT_ALPHA so step 0 matches the paper.
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.zeros_(self.hyper.weight)
        nn.init.constant_(self.hyper.bias, self.INIT_ALPHA)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        rec_repr_space = self._rec_view(x)                                     # P~
        original_repr_space = self._origin_view(x)                             # P
        e_orig = self.value_embedding_org(original_repr_space)                 # E^c (eq. 12)
        e_rec = self.value_embedding_rec(rec_repr_space)                       # E^s (eq. 12)
        # Context: mean across batch+channels and patches of E^c -> [d_model]
        ctx = e_orig.mean(dim=0).mean(dim=0)
        h = torch.relu(self.context_proj(ctx))                                 # [HYPER_HIDDEN]
        alpha_dyn = self.hyper(h).view(self.patch_num, e_orig.shape[-1])       # [n, d_model]
        weight = torch.sigmoid(alpha_dyn)                                      # alpha in (0,1)
        embedding = weight * e_orig + (1.0 - weight) * e_rec                   # eq. 13
        if self.pos:
            embedding = embedding + self.position_embedding(original_repr_space)
        return self.dropout(embedding), n_vars


# ---------------------------------------------------------------------------
# Factorial combinations: (any _select variant) x (Hypernet alpha fusion)
# ---------------------------------------------------------------------------
def _make_hypernet_combo(base_select_cls, combo_name):
    """Factory: compose a base SRS variant's ``_select`` with HypernetAF fusion.

    WHY: every (Random/TASP) x HypernetAF combo shares identical fusion code;
    we keep it in one place. Only the inherited ``_select`` differs.
    """

    class _Combo(base_select_cls):
        """Factorial combination: <base_select_cls._select> + Hypernet-AF."""

        HYPER_HIDDEN = SRSHypernetAF.HYPER_HIDDEN
        INIT_ALPHA = SRSHypernetAF.INIT_ALPHA

        def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                     hidden_size, alpha=2.0, pos=True, **extra_kwargs):
            super().__init__(d_model, patch_len, stride, seq_len, dropout,
                             hidden_size, alpha, pos, **extra_kwargs)
            # Same zero-init hypernet pattern as SRSHypernetAF.
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
            # _rec_view calls the inherited _select (random / engineered),
            # which is what makes each combo different.
            rec_repr_space = self._rec_view(x)
            original_repr_space = self._origin_view(x)
            e_orig = self.value_embedding_org(original_repr_space)
            e_rec = self.value_embedding_rec(rec_repr_space)
            ctx = e_orig.mean(dim=0).mean(dim=0)                             # context vector [d_model]
            h = torch.relu(self.context_proj(ctx))
            alpha_dyn = self.hyper(h).view(self.patch_num, e_orig.shape[-1])
            weight = torch.sigmoid(alpha_dyn)
            embedding = weight * e_orig + (1.0 - weight) * e_rec             # data-dependent fusion
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


# ---------------------------------------------------------------------------
# Three-extension combos: <base _select> + identity _shuffle + Hypernet-AF
# ---------------------------------------------------------------------------
def _make_3way_combo(two_way_combo_cls, combo_name):
    """Factory: take an existing 2-axis combo and add identity ``_shuffle``.

    WHY: 3-axis change vs vanilla SRS  --  Select != Learned, Shuffle == Identity,
    Fusion == Hypernet alpha. Used to ask: "is the learned shuffle doing anything
    the 2-axis combos overlooked?"
    """

    class _ThreeWayCombo(two_way_combo_cls):
        """``two_way_combo_cls`` + identity shuffle (drops the learned reorder)."""

        def _shuffle(self, selected_patches):
            return selected_patches                                          # identity

    _ThreeWayCombo.__name__ = combo_name
    _ThreeWayCombo.__qualname__ = combo_name
    return _ThreeWayCombo


SRSTimeAware_NoShuffle_HypernetAF = _make_3way_combo(
    SRSTimeAware_HypernetAF, "SRSTimeAware_NoShuffle_HypernetAF"
)
SRSRandomSP_NoShuffle_HypernetAF = _make_3way_combo(
    SRSRandomSP_HypernetAF, "SRSRandomSP_NoShuffle_HypernetAF"
)


# ---------------------------------------------------------------------------
# SRSNetModel subclasses (swap the patch_embedding for the random variant)
# ---------------------------------------------------------------------------
class _SelectivityControlsSRSNetModel(SRSNetModel):
    """SRSNetModel base that swaps the patch_embedding for one of our SRS variants."""

    embedding_cls = None  # override in subclass with an SRS variant

    def __init__(self, config):
        super().__init__(config)                                             # builds revin, default SRS, head
        if self.embedding_cls is None:
            raise NotImplementedError("Subclass must set embedding_cls to an SRS variant.")
        # Same constructor signature as the parent SRS class so we can hot-swap.
        kwargs = dict(
            d_model=config.d_model, patch_len=self.patch_len, stride=self.stride,
            seq_len=self.seq_len, dropout=config.dropout, hidden_size=config.hidden_size,
            alpha=config.alpha, pos=config.pos,
        )
        self.patch_embedding = self.embedding_cls(**kwargs)                  # replace SRS with our variant


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


# Factorial combinations
class SRSNet_TASP_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSTimeAware_HypernetAF


class SRSNet_RandomSP_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSP_HypernetAF


# Three-extension combos
class SRSNet_TASP_NoShuffle_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSTimeAware_NoShuffle_HypernetAF


class SRSNet_RandomSP_NoShuffle_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSP_NoShuffle_HypernetAF


# ---------------------------------------------------------------------------
# DeepForecastingModelBase wrappers (registered in __init__.py)
# ---------------------------------------------------------------------------
class _SelectivityControlsSRSNet(SRSNet):
    """SRSNet TFB wrapper that reports a distinct model_name + model_cls per variant."""

    variant_name = None  # override: string used in result CSVs
    model_cls = None     # override: the _SelectivityControlsSRSNetModel subclass

    @property
    def model_name(self):
        """String TFB uses to label this variant in result CSVs."""
        return self.variant_name

    def _init_model(self):
        """Instantiate the underlying nn.Module from the merged config."""
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


# Factorial combinations (Select x Fusion)
class SRSNet_TASP_HypernetAF(_SelectivityControlsSRSNet):
    """Engineered-feature select + Hypernet alpha fusion."""

    variant_name = "SRSNet_TASP_HypernetAF"
    model_cls = SRSNet_TASP_HypernetAF_Model


class SRSNet_RandomSP_HypernetAF(_SelectivityControlsSRSNet):
    """Random select + Hypernet alpha fusion."""

    variant_name = "SRSNet_RandomSP_HypernetAF"
    model_cls = SRSNet_RandomSP_HypernetAF_Model


# Three-extension combos (Select != Learned + Identity shuffle + Hypernet alpha)
class SRSNet_TASP_NoShuffle_HypernetAF(_SelectivityControlsSRSNet):
    """Engineered-feature select + identity shuffle + Hypernet alpha fusion."""

    variant_name = "SRSNet_TASP_NoShuffle_HypernetAF"
    model_cls = SRSNet_TASP_NoShuffle_HypernetAF_Model


class SRSNet_RandomSP_NoShuffle_HypernetAF(_SelectivityControlsSRSNet):
    """Random select + identity shuffle + Hypernet alpha fusion."""

    variant_name = "SRSNet_RandomSP_NoShuffle_HypernetAF"
    model_cls = SRSNet_RandomSP_NoShuffle_HypernetAF_Model


# ---------------------------------------------------------------------------
# Backbone extension: SRS + TransformerEncoder + FlattenHead
# ---------------------------------------------------------------------------
# Tests the paper's claim (Sec. 3.4 + Sec. 4) that a linear head over the
# SRS embeddings is enough. Inserts a real Transformer Encoder between
# the SRS output and the linear head and lets us measure the delta.
#
# Two extra hyper-params (read from config via getattr, defaults provided):
#   encoder_n_heads  : number of self-attention heads (default 8)
#   encoder_n_layers : number of Transformer encoder layers (default 2)


class _SRSNetWithEncoderModel(SRSNetModel):
    """SRSNet baseline + Transformer Encoder between patch_embedding and head.

    Same RevIN + SRS + FlattenHead as the paper, just with an encoder
    block inserted on the SRS embeddings before they reach the head.
    """

    def __init__(self, config):
        super().__init__(config)                                                # builds revin, patch_embedding, head
        n_heads = getattr(config, "encoder_n_heads", 8)
        n_layers = getattr(config, "encoder_n_layers", 2)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)    # [B*N, n, d] -> [B*N, n, d]

    def forward(self, x_enc):
        # x_enc: [B, T, N]
        x_enc = self.revin(x_enc, 'norm')                                       # [B, T, N]    -- Sec. 3.1 norm
        x_enc = x_enc.permute(0, 2, 1)                                          # [B, N, T]    -- channel-independent layout

        enc_out, n_vars = self.patch_embedding(x_enc)                           # [B*N, n, d]  -- SRS module (Sec. 3.2-3.4)
        enc_out = self.encoder(enc_out)                                         # [B*N, n, d]  -- self-attention over patches  ← NEW

        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)                                   # [B, N, d, n] -- FlattenHead wants d before n

        dec_out = self.head(enc_out).permute(0, 2, 1)                           # [B, L, N]    -- forecast in normalized scale
        return self.revin(dec_out, 'denorm')                                    # [B, L, N]    -- back to original scale


class _SelectivityControlsSRSNetEncoderModel(_SelectivityControlsSRSNetModel):
    """Selectivity-controls model (TASP / HypernetAF / ...) + Transformer Encoder.

    Combines the 'swap patch_embedding' machinery of
    ``_SelectivityControlsSRSNetModel`` with the Transformer Encoder addition
    of ``_SRSNetWithEncoderModel``.
    """

    def __init__(self, config):
        super().__init__(config)                                                # swaps patch_embedding via embedding_cls
        n_heads = getattr(config, "encoder_n_heads", 8)
        n_layers = getattr(config, "encoder_n_layers", 2)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x_enc):
        # Same shape pipeline as _SRSNetWithEncoderModel, with the swapped patch_embedding
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


# --- TFB wrappers for the encoder variants -----------------------------------
class SRSNet_TransformerEncoder(_SelectivityControlsSRSNet):
    """Baseline SRSNet + Transformer Encoder. Tests 'linear head is enough'."""

    variant_name = "SRSNet_TransformerEncoder"
    model_cls = _SRSNetWithEncoderModel


class SRSNet_TASP_TransformerEncoder(_SelectivityControlsSRSNet):
    """TASP select + Transformer Encoder (combo across Select x Backbone)."""

    variant_name = "SRSNet_TASP_TransformerEncoder"
    model_cls = SRSNet_TASP_TransformerEncoder_Model


class SRSNet_HypernetAF_TransformerEncoder(_SelectivityControlsSRSNet):
    """Hypernet-AF fusion + Transformer Encoder (combo across Fusion x Backbone)."""

    variant_name = "SRSNet_HypernetAF_TransformerEncoder"
    model_cls = SRSNet_HypernetAF_TransformerEncoder_Model


# Backwards compatibility alias: the old broad-extensions code referenced
# ``srs_paper.SRSNet_RandomSRS``.  Keep that name as an alias so any stale
# tasks in older manifests still resolve, but emit no behaviour difference.
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
    # Backbone extension (Transformer Encoder)
    SRSNet_TransformerEncoder,
    SRSNet_TASP_TransformerEncoder,
    SRSNet_HypernetAF_TransformerEncoder,
):
    _cls.MODEL_HYPER_PARAMS = MODEL_HYPER_PARAMS


__all__ = [
    # Layers -- random controls
    "SRSRandomSP",
    "SRSRandomSPNoShuffle",
    "SRSRandomSPRandomShuffle",
    # Layers -- constructive extensions
    "SRSTimeAware",
    "SRSHypernetAF",
    # Layers -- factorial combinations (2-axis)
    "SRSTimeAware_HypernetAF",
    "SRSRandomSP_HypernetAF",
    # Layers -- 3-axis combinations (identity shuffle on top of 2-axis)
    "SRSTimeAware_NoShuffle_HypernetAF",
    "SRSRandomSP_NoShuffle_HypernetAF",
    # Model wrappers
    "SRSNet_RandomSP",
    "SRSNet_RandomSPNoShuffle",
    "SRSNet_RandomSPRandomShuffle",
    "SRSNet_TASP",
    "SRSNet_HypernetAF",
    "SRSNet_TASP_HypernetAF",
    "SRSNet_RandomSP_HypernetAF",
    "SRSNet_TASP_NoShuffle_HypernetAF",
    "SRSNet_RandomSP_NoShuffle_HypernetAF",
    # Backbone extension (Transformer Encoder)
    "SRSNet_TransformerEncoder",
    "SRSNet_TASP_TransformerEncoder",
    "SRSNet_HypernetAF_TransformerEncoder",
    # Backwards-compat alias
    "SRSNet_RandomSRS",
]
