"""Selectivity-controls extensions for SRSNet reproduction.

This module replaces the earlier "broad extension set" (DDA / MSP / GAF /
RandomSRS) with a focused selectivity-controls study, per the plan in
``scripts/repro/selectivity_extension_plan.md``.

Rationale (summary -- read the plan for the full discussion):

    * The earlier extensions were either capacity-confounded (GAF, MSP),
      effectively no-ops in the sigmoid-saturated regime (DDA), or imprecisely
      named and entangled with both selection and shuffling (RandomSRS).
    * The selectivity-controls study tests one sharp question instead:
        How much does SRSNet's performance depend on the **learned** patch
        selector, compared with random patch choices?
    * Two retrained controls are added.  Both share the SRSNet architecture
      except for the SRS ``_select`` (and, in the third variant, the SRS
      ``_shuffle``):

        * ``SRSRandomSP``                  -- random ``_select`` only,
                                              learned ``_shuffle`` kept.
        * ``SRSRandomSPNoShuffle``         -- random ``_select`` + identity
                                              shuffle (deterministic).  This
                                              is the **default** second
                                              variant per review note #3 of
                                              the plan.
        * ``SRSRandomSPRandomShuffle``     -- random ``_select`` + random
                                              ``_shuffle``.  Optional, kept
                                              for ablating both stages at
                                              once.

For naming hygiene (review note #6) the old ``SRSNet_RandomSRS`` wrapper is
**aliased** to ``SRSNet_RandomSP`` so old call-sites do not silently break,
but new code should use the precise name.
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
    """SRS variant that selects candidate windows uniformly at random.

    Everything else (``_shuffle``, adaptive fusion, embeddings, dropout,
    positional handling) is inherited unchanged from the parent ``SRS``.
    The only intervention is in ``_select``: instead of scoring each
    candidate window through the learned ``scorer_select`` MLP, we draw
    a uniform random index for each output slot.

    Two consequences worth noting:

    * The learned ``scorer_select`` parameters still exist (so the
      state-dict shape matches a vanilla SRSNet), but they receive zero
      gradient through the forward pass.  This is intentional -- the goal
      is a tight ``selectivity claim'' control, not a parameter-count
      ablation.
    * The original ``_select`` multiplies each selected patch by a unit-
      scaled ``max_scores`` factor.  Random selection has no scores, so we
      drop that multiplication.  This matches the sketch in the plan.
    """

    def _select(self, x_rec):
        # x_rec shape: [batch, n_vars, candidate_num, patch_size]
        batch, n_vars, candidate_num, patch_size = x_rec.shape
        idx = torch.randint(
            low=0,
            high=candidate_num,
            size=(batch, n_vars, 1, self.patch_num),
            device=x_rec.device,
        )
        # Broadcast indices over the patch_size dim so torch.gather can
        # collect the entire patch window for each chosen candidate.
        gather_idx = idx.repeat(1, 1, patch_size, 1).permute(0, 1, 3, 2)
        return torch.gather(x_rec, dim=-2, index=gather_idx)


class SRSRandomSPNoShuffle(SRSRandomSP):
    """Random ``_select`` + identity ``_shuffle`` (deterministic shuffle).

    Per review note #3 of the plan, identity shuffle is the recommended
    default for the second variant because it introduces no extra source
    of stochasticity and isolates "no learning in ``_shuffle``" cleanly.
    """

    def _shuffle(self, selected_patches):
        return selected_patches


class SRSRandomSPRandomShuffle(SRSRandomSP):
    """Random ``_select`` + random ``_shuffle`` (full stochastic baseline).

    Kept as an optional third variant for users who want to ablate both
    selection and shuffling at the same time.  Has higher per-seed
    variance than ``SRSRandomSPNoShuffle`` because both stages are now
    sampled fresh each forward pass.
    """

    def _shuffle(self, selected_patches):
        batch, n_vars, patch_num, patch_size = selected_patches.shape
        scores = torch.rand(
            batch, n_vars, patch_num, 1, device=selected_patches.device
        )
        order = torch.argsort(scores, dim=-2, descending=True)
        gather_idx = order.repeat(1, 1, 1, patch_size)
        return torch.gather(selected_patches, dim=-2, index=gather_idx)


# ---------------------------------------------------------------------------
# Constructive extensions: TASP, Hypernet-AF
# ---------------------------------------------------------------------------
class SRSTimeAware(SRS):
    """TASP: scorer over engineered interpretable features.

    Addresses paper FW#1 (environment-aware mechanism) + L3 (interpretability).
    Replaces the [patch_len -> hidden -> patch_num] MLP scorer with a
    [4 -> 16 -> patch_num] MLP whose inputs are 4 per-window summary
    statistics: dominant FFT magnitude, lag-1 autocorrelation, variance,
    and trend slope.  Total scorer parameters drop sharply, so any
    improvement cannot be attributed to extra capacity.
    """

    N_FEATURES = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the learned scorer over raw patch values with a tiny
        # MLP over engineered per-window features.  The shuffle scorer
        # is left alone so any change in MSE is attributable to the
        # selection scorer alone.
        hidden = max(self.N_FEATURES * 4, 16)
        self.scorer_select = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.patch_num),
        )

    @staticmethod
    def _window_features(x_rec):
        """x_rec: [B, C, candidate_num, patch_size] -> [B, C, candidate_num, 4]."""
        # Dominant non-DC FFT magnitude per window.
        spec = torch.fft.rfft(x_rec, dim=-1).abs()
        if spec.shape[-1] > 1:
            dominant = spec[..., 1:].max(dim=-1).values
        else:
            dominant = spec[..., 0]
        # Demean for autocorrelation and trend-slope computation.
        mean = x_rec.mean(dim=-1, keepdim=True)
        var = x_rec.var(dim=-1, unbiased=False)
        x_c = x_rec - mean
        # Lag-1 autocorrelation, biased toward zero for very short windows.
        denom = x_c.pow(2).sum(dim=-1) + 1e-8
        ac1 = (x_c[..., 1:] * x_c[..., :-1]).sum(dim=-1) / denom
        # Trend slope: least-squares fit residual.
        t = torch.arange(x_rec.shape[-1], device=x_rec.device, dtype=x_rec.dtype)
        t_c = t - t.mean()
        slope_denom = (t_c * t_c).sum() + 1e-8
        slope = ((x_rec - mean) * t_c).sum(dim=-1) / slope_denom
        feats = torch.stack([dominant, ac1, var, slope], dim=-1)
        return feats

    def _select(self, x_rec):
        # x_rec: [B, C, candidate_num, patch_size]
        feats = self._window_features(x_rec)
        scores = self.scorer_select(feats)  # [B, C, candidate_num, patch_num]
        indices = torch.argmax(scores, dim=-2, keepdim=True)
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)
        max_scores[non_zero_mask] *= inv
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches
        return selected_patches


class SRSHypernetAF(SRS):
    """Hypernet-AF: data-dependent alpha via a tiny hypernet.

    Addresses paper FW#3 (efficient alpha update mechanism) + L4
    (initialization).  Replaces the free [patch_num, d_model] alpha
    parameter with a small hypernet that consumes a batch-level context
    vector and outputs the per-cell alpha.  Zero-init of the hypernet
    weights plus a constant bias of 3.5 makes step 0 behave identically
    to the paper's default (sigmoid(3.5) ~ 0.97), so any deviation must
    come from learned context-dependence rather than capacity at init.
    """

    HYPER_HIDDEN = 8
    INIT_ALPHA = 3.5

    def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                 hidden_size, alpha=2.0, pos=True):
        super().__init__(d_model, patch_len, stride, seq_len, dropout,
                         hidden_size, alpha, pos)
        # Remove the free alpha parameter; the hypernet replaces it.
        # We still keep the buffer name for state-dict shape compatibility
        # with vanilla SRSNet, but it will not be used in forward().
        if hasattr(self, "alpha"):
            self.register_buffer("_unused_alpha", torch.zeros_like(self.alpha.data))
            del self.alpha
        self.context_proj = nn.Linear(d_model, self.HYPER_HIDDEN)
        self.hyper = nn.Linear(self.HYPER_HIDDEN, self.patch_num * d_model)
        # Vanilla-preserving init: zero weights + bias = INIT_ALPHA on the
        # output so sigmoid(alpha_t=0) ~ 0.97, matching the paper baseline.
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
        # Context: mean across batch*nvars and across patches of e_orig.
        # Shape after the SRS forward of the parent: [bs*nvars, patch_num, d_model].
        ctx = e_orig.mean(dim=0).mean(dim=0)  # [d_model]
        h = torch.relu(self.context_proj(ctx))
        alpha_dyn = self.hyper(h).view(self.patch_num, e_orig.shape[-1])
        weight = torch.sigmoid(alpha_dyn)
        embedding = weight * e_orig + (1.0 - weight) * e_rec
        if self.pos:
            embedding = embedding + self.position_embedding(original_repr_space)
        return self.dropout(embedding), n_vars


# ---------------------------------------------------------------------------
# Factorial combinations: (any _select variant) x (Hypernet alpha fusion)
# ---------------------------------------------------------------------------
def _make_hypernet_combo(base_select_cls, combo_name):
    """Compose a base SRS variant (which defines ``_select``) with Hypernet
    alpha fusion (which overrides ``forward`` to replace the free alpha
    parameter with a tiny hypernet over a batch-level context vector).

    The factory keeps the ``__init__`` and ``forward`` body in one place so
    that all (Random/TASP) x HypernetAF combos share identical fusion
    code.  Only the inherited ``_select`` differs.
    """

    class _Combo(base_select_cls):
        """Factorial combination: <base_select_cls._select> + Hypernet-AF."""

        HYPER_HIDDEN = SRSHypernetAF.HYPER_HIDDEN
        INIT_ALPHA = SRSHypernetAF.INIT_ALPHA

        def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                     hidden_size, alpha=2.0, pos=True, **extra_kwargs):
            super().__init__(
                d_model, patch_len, stride, seq_len, dropout,
                hidden_size, alpha, pos, **extra_kwargs,
            )
            # Replace the free alpha parameter with the same zero-init
            # hypernet pattern used by SRSHypernetAF (vanilla-preserving).
            if hasattr(self, "alpha"):
                self.register_buffer(
                    "_unused_alpha", torch.zeros_like(self.alpha.data)
                )
                del self.alpha
            self.context_proj = nn.Linear(d_model, self.HYPER_HIDDEN)
            self.hyper = nn.Linear(
                self.HYPER_HIDDEN, self.patch_num * d_model
            )
            nn.init.zeros_(self.context_proj.weight)
            nn.init.zeros_(self.context_proj.bias)
            nn.init.zeros_(self.hyper.weight)
            nn.init.constant_(self.hyper.bias, self.INIT_ALPHA)

        def forward(self, x):
            n_vars = x.shape[1]
            x = self.padding_patch_layer(x)
            # _rec_view calls _select internally; the inherited _select
            # (random / engineered / supervised) is what makes each combo
            # different.
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
                embedding = embedding + self.position_embedding(
                    original_repr_space
                )
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
    """Add an identity ``_shuffle`` on top of an existing 2-way combo.

    Each input ``two_way_combo_cls`` already pairs a custom ``_select``
    with Hypernet-AF fusion.  This factory subclasses it once more to
    drop the learned shuffle as well, producing a 3-axis change versus
    vanilla SRS:

        Select != Learned   AND   Shuffle == Identity   AND
        Fusion == Hypernet alpha

    Used to ask: "is the learned shuffle doing anything that the
    factorial combos overlooked?"  Identity shuffle was chosen rather
    than random shuffle because review note #3 of the original
    selectivity plan picks it as the cleanest default control.
    """

    class _ThreeWayCombo(two_way_combo_cls):
        """``two_way_combo_cls`` semantics + identity shuffle."""

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


# ---------------------------------------------------------------------------
# SRSNetModel subclasses (swap the patch_embedding for the random variant)
# ---------------------------------------------------------------------------
class _SelectivityControlsSRSNetModel(SRSNetModel):
    """SRSNetModel that swaps the patch_embedding for a control layer."""

    embedding_cls = None  # override in subclass

    def __init__(self, config):
        super().__init__(config)
        if self.embedding_cls is None:
            raise NotImplementedError(
                "Subclass must set embedding_cls to an SRS variant."
            )
        kwargs = dict(
            d_model=config.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            seq_len=self.seq_len,
            dropout=config.dropout,
            hidden_size=config.hidden_size,
            alpha=config.alpha,
            pos=config.pos,
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
    """SRSNet variant that reports a distinct model_name + model class."""

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
