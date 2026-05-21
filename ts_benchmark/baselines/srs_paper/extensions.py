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
    """Random ``_select``, learned ``_shuffle`` -- negative control for the scorer.

    ----------------------------------------------------------------
    1) The problem this control tries to expose
    ----------------------------------------------------------------
    Vanilla SRS uses Scorer^s + Argmax to pick the n "best" candidates
    out of K. The paper claims this finds the most informative
    subsequences, but the claim is not directly tested. We have no way
    to know if the scorer matters, or if any n patches would do.

    ----------------------------------------------------------------
    2) The idea
    ----------------------------------------------------------------
    Replace Scorer^s + Argmax with a uniform random index per output
    slot. Everything else (learned shuffle, fusion, embedding) stays
    untouched. If random selection matches the learned one in MSE,
    the "selectivity" claim does not hold on our grid.
    """

    def _select(self, x_rec):
        """Random selection: skip Scorer^s + Argmax, draw uniform indices instead.

        Paper notation (only the symbols that survive in the random control):
            P'        all K stride=1 candidate patches (input)
            idx       uniform random indices in [0, K)      (replaces I^s)
            P^s_max   the n picked patches (returned, no STE)
        """
        # P' = x_rec  [B, N, K, p]
        batch, n_vars, candidate_num, patch_size = x_rec.shape

        # Replaces eq. 1+2: draw uniform random index per (batch, channel, slot).
        # No Scorer^s call -- scorer_select still exists for state-dict compat
        # but receives no gradient through this forward.
        idx = torch.randint(low=0, high=candidate_num,
                            size=(batch, n_vars, 1, self.patch_num),
                            device=x_rec.device)                            # idx   [B, N, 1, n]

        # Reshape to the [B, N, n, p] form that torch.gather expects:
        # repeat the index across patch_len so we pull a whole patch per slot.
        gather_idx = idx.repeat(1, 1, patch_size, 1).permute(0, 1, 3, 2)    # gather_idx   [B, N, n, p]

        # Replaces eq. 4 (and skips eq. 3+5 entirely -- no STE needed):
        # P^s_max = P'[idx]
        return torch.gather(x_rec, dim=-2, index=gather_idx)                # P^s_max   [B, N, n, p]


class SRSRandomSPNoShuffle(SRSRandomSP):
    """Random ``_select`` + identity ``_shuffle`` -- no learning anywhere in SRS.

    ----------------------------------------------------------------
    1) The problem this control tries to expose
    ----------------------------------------------------------------
    Even after SRSRandomSP removes the learned selector, the learned
    Scorer^r is still active and could be carrying SRS on its own. If
    the model still works with both stages disabled, then SRS itself
    adds nothing and the paper's contribution reduces to "patch + head".

    ----------------------------------------------------------------
    2) The idea
    ----------------------------------------------------------------
    Inherit the random _select from SRSRandomSP and override _shuffle
    to be the identity. No learned MLP runs inside the SRS path at all:
    cleanest "SRS does nothing" baseline.
    """

    def _shuffle(self, selected_patches):
        """Identity reordering: skip Scorer^r + Argsort entirely.

        Paper notation:
            P~^s_max  the n selected patches (input, from _select)
            P~        P~^s_max unchanged (returned)
        """
        # P~ = P~^s_max  [B, N, n, p]   (no reorder)
        return selected_patches


class SRSRandomSPRandomShuffle(SRSRandomSP):
    """Random ``_select`` + random ``_shuffle`` -- noise upper bound for SRS.

    ----------------------------------------------------------------
    1) The problem this control tries to expose
    ----------------------------------------------------------------
    NoShuffle keeps patches in selection order, which is still a specific
    (not arbitrary) ordering. The question: does ANY structure in the
    SRS path matter, or is the model robust to pure noise on both stages?

    ----------------------------------------------------------------
    2) The idea
    ----------------------------------------------------------------
    Random _select (inherited) + random _shuffle (uniform scores ->
    argsort). Both stages re-sample every forward pass. If the model
    still works, the patches themselves carry the signal regardless
    of order; if it crashes, even random "structure" beats no structure.
    """

    def _shuffle(self, selected_patches):
        """Random reordering: skip Scorer^r + Argsort, use random scores instead.

        Paper notation (only the symbols that survive in the random control):
            P~^s_max  the n selected patches (input, from _select)
            scores    uniform random in [0, 1)           (replaces S^r)
            order     argsort(scores)                    (replaces I^r)
            P~        randomly permuted patches (returned, no STE)
        """
        # P~^s_max = selected_patches  [B, N, n, p]
        batch, n_vars, patch_num, patch_size = selected_patches.shape

        # Replaces eq. 6: uniform random scores instead of Scorer^r output.
        scores = torch.rand(batch, n_vars, patch_num, 1,
                            device=selected_patches.device)                 # scores   [B, N, n, 1]

        # Replaces eq. 7: argsort of random scores = uniform random permutation.
        # E.g. scores=[0.3, 0.9, 0.1] -> order=[1, 0, 2].
        order = torch.argsort(scores, dim=-2, descending=True)              # order   [B, N, n, 1]

        # Broadcast indices along patch_len so we reorder whole patches.
        gather_idx = order.repeat(1, 1, 1, patch_size)                      # gather_idx   [B, N, n, p]

        # Replaces eq. 9 (and skips eq. 8+10 -- no STE needed):
        # P~ = P^r_sort = P~^s_max[order]
        return torch.gather(selected_patches, dim=-2, index=gather_idx)     # P~   [B, N, n, p]


# ---------------------------------------------------------------------------
# Constructive extensions: TASP, Hypernet-AF
# ---------------------------------------------------------------------------
class SRSTimeAware(SRS):
    """TASP -- Time-Aware Selective Patching (paper FW#1 + L3).

    ----------------------------------------------------------------
    1) The problem TASP tries to solve
    ----------------------------------------------------------------
    Vanilla SRS uses this scorer:

        Linear(patch_len, hidden) -> ReLU -> Linear(hidden, patch_num)
          24  ->  128                          128 -> n

    An MLP that eats 24 raw patch values and produces n scores. It is
    a black box: we have no idea WHY the model picks certain patches.
    The paper claims "selective patching adaptively selects informative
    subsequences" but the claim is not verifiable: the MLP could be
    learning anything.

    The paper itself acknowledges this in Section 6:
        * FW#1: "develop time-series-specific selection mechanisms based on environmental priors" (use domain knowledge)
        * L3:   the model is not interpretable

    TASP is our answer to both points.

    ----------------------------------------------------------------
    2) The TASP idea
    ----------------------------------------------------------------
    Instead of feeding the scorer 24 raw values, we feed it 4
    interpretable signal-processing features computed per patch:

        #  Feature                     Answers...
        -- --------------------------- --------------------------------
        1  Dominant FFT magnitude      How periodic is this window?
        2  Lag-1 autocorrelation       How smooth vs noisy?
        3  Variance                    How much does it fluctuate?
        4  Trend slope                 Going up / flat / down?

    The MLP shrinks dramatically:

                          Vanilla        TASP
        Input             24 (raw)       4  (engineered)
        Hidden            128            16
        Output            n              n
        Params total     ~6,000         ~370   (-94%)

    Key point: if TASP matches or beats vanilla with 94% fewer scorer
    parameters, those 4 features carry enough information to pick
    patches well AND the choice is interpretable (e.g. "the model
    picked the most periodic patch", "the one with the strongest
    trend"). The shuffle scorer is left intact, so any MSE change is
    attributable only to _select.
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
        """Compute feature (FFT, lag-1 autocorr, var, trend slope) per window.

        x_rec: [B, C, candidate_num, patch_size] -> [B, C, candidate_num, 4]
        """
        # Feature 1: Dominant non-DC FFT magnitude 
        # Apply an FFT to break the patch into sine waves.
        # Reading: high = the patch has a strong periodic component; low = flat or aperiodic.
        spec = torch.fft.rfft(x_rec, dim=-1).abs()
        dominant = spec[..., 1:].max(dim=-1).values if spec.shape[-1] > 1 else spec[..., 0]

        # Shared prep for the next two features: patch mean + centered signal.
        mean = x_rec.mean(dim=-1, keepdim=True)
        x_c = x_rec - mean

        # Feature 3: Variance 
        # Plain variance: how much the values bounce around their own mean.
        # Reading: high = the patch fluctuates a lot; 
        #          low = flat / stable.
        var = x_rec.var(dim=-1, unbiased=False)

        # Feature 2: Lag-1 autocorrelation 
        # Correlation between the signal and itself shifted by one step.
        # Closed form on the centered signal:
        #     ac1 = sum( x_c[t] * x_c[t-1] )  /  sum( x_c[t]^2 )
        # Reading: +1 = very smooth (neighbour values almost identical);
        #          0 = noisy, like white noise;
        #          -1 = oscillating / zigzag.
        denom = x_c.pow(2).sum(dim=-1) + 1e-8
        ac1 = (x_c[..., 1:] * x_c[..., :-1]).sum(dim=-1) / denom

        # Feature 4: Trend slope 
        # Slope of the best-fit straight line through the patch
        # Reading: > 0 = patch is going up;
        #          0 = flat;
        #          < 0 = going down.
        t = torch.arange(x_rec.shape[-1], device=x_rec.device, dtype=x_rec.dtype)
        t_c = t - t.mean()
        slope_denom = (t_c * t_c).sum() + 1e-8
        slope = ((x_rec - mean) * t_c).sum(dim=-1) / slope_denom

        # Stack the four features along a new axis -> [B, C, K, 4]
        return torch.stack([dominant, ac1, var, slope], dim=-1)

    def _select(self, x_rec):
        """Selective Patching with engineered features (paper Sec. 3.2, TASP variant).

        Same flow as vanilla SRS._select; only the scorer input changes:
        4 engineered features (FFT max, ac1, var, slope) instead of raw
        patch values. From eq. 1 onward everything matches the paper.

        Paper notation (same as vanilla):
            P'        all K stride=1 candidate patches  (input)
            Scorer^s  MLP that scores patches (TASP version: over 4 features)
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

        # NEW vs vanilla SRS: replace raw values with 4 interpretable features.
        feats = self._window_features(x_rec)                                # [B, N, K, 4]   (FFT max, ac1, var, slope)

        # eq. 1+2:  S^s = Scorer^s(features),   I^s = Argmax_K(S^s)
        scores  = self.scorer_select(feats)                                 # S^s   [B, N, K, n]
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


class SRSHypernetAF(SRS):
    """Hypernet-AF -- data-dependent Adaptive Fusion (paper FW#3 + L4).

    ----------------------------------------------------------------
    1) The problem Hypernet-AF tries to solve
    ----------------------------------------------------------------
    Vanilla SRS fuses the two views with one static parameter:

        alpha = nn.Parameter([n, d_model])   # learned once, same for every input

    At forward time the same alpha gets reused for every sample in the
    batch : the fusion weight does not depend on what the input actually
    looks like. The paper itself acknowledges this in Section 6:
        * FW#3: "develop a more efficient update mechanism for alpha"
        * L4:   the alpha init is a free hyper-parameter (no learned signal)

    Hypernet-AF is our answer: make alpha a function of the input.

    ----------------------------------------------------------------
    2) The Hypernet-AF idea
    ----------------------------------------------------------------
    Instead of one static [n, d_model] alpha, we produce alpha per-batch
    from a tiny hypernet over a context vector summarising the batch:

        ctx = mean over batch * channels and over patches of E^c   (one vector of size d_model)
        h   = ReLU(W_proj  @ ctx + b_proj)                          (HYPER_HIDDEN = 8)
        a   = W_hyper @ h + b_hyper                                  (flat alpha, n*d_model entries)
        alpha_dyn = a.view(n, d_model)
        weight   = sigmoid(alpha_dyn)

    Two important init choices keep the model "vanilla-preserving" at step 0:
        * W_proj, W_hyper, b_proj  : zero-initialised
        * b_hyper                  : constant = INIT_ALPHA = 3.5
    so at step 0 alpha_dyn = 3.5 for every cell, sigmoid(3.5) ~= 0.97 --
    the same value the original SRS uses by default. Any deviation from
    this baseline must come from the hypernet LEARNING to make alpha
    context-dependent, not from extra capacity at init.

    The free [n, d_model] alpha parameter is replaced by a non-trainable
    buffer of the same shape so state-dict shapes stay compatible with
    vanilla SRSNet checkpoints.
    """

    HYPER_HIDDEN = 8
    INIT_ALPHA = 3.5

    def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                 hidden_size, alpha=2.0, pos=True):
        """Build SRS as usual, then swap the free alpha for the hypernet.

        Paper notation -> code attribute:
            alpha             self.alpha (vanilla) -> replaced
            context_proj      W_proj  (Linear: d_model -> HYPER_HIDDEN)
            hyper             W_hyper (Linear: HYPER_HIDDEN -> n*d_model)
            INIT_ALPHA = 3.5  bias of the hypernet output, vanilla-preserving init
        """
        super().__init__(d_model, patch_len, stride, seq_len, dropout,
                         hidden_size, alpha, pos)

        # Drop the free alpha; keep a non-trainable buffer of the same shape with all zeros so a vanilla SRSNet state-dict still loads without shape errors.
        if hasattr(self, "alpha"):
            self.register_buffer("_unused_alpha", torch.zeros_like(self.alpha.data))
            del self.alpha

        # Hypernet: ctx -> h -> flat alpha
        self.context_proj = nn.Linear(d_model, self.HYPER_HIDDEN)                   # shrink: d_model -> H
        self.hyper        = nn.Linear(self.HYPER_HIDDEN, self.patch_num * d_model)  # expand: H -> n*d_model

        # Vanilla-preserving init: zero weights everywhere + bias = INIT_ALPHA on
        # the output layer, so at step 0 we have sigmoid(INIT_ALPHA) ~= 0.97 --
        # the same fusion weight as default SRSNet.
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.zeros_(self.hyper.weight)
        nn.init.constant_(self.hyper.bias, self.INIT_ALPHA)

    def forward(self, x):
        """SRS forward with Hypernet-AF fusion in place of the static alpha.

        Paper notation:
            X         input lookback window
            X'        right-padded X
            P         original view  (adjacent patches)
            P~        reconstructive view  (selective + shuffled)
            E^c       PatchEmbedding^1(P)
            E^s       PatchEmbedding^2(P~)
            ctx       mean-pooled context vector                                 [d_model]   (NEW)
            h         hypernet hidden activation                                 [H]         (NEW)
            alpha_dyn per-instance fusion weight (replaces the static alpha)     [n, d]      (NEW)
            alpha     sigmoid(alpha_dyn) in (0, 1)                               [n, d]      (NEW)
            E~        convex combo of E^c and E^s
            E         E~ + PositionEmbedding(P)  (returned)

        Shapes: B=batch, N=channels, T=lookback, T_pad=padded length,
                n=patch_num, p=patch_len, d=d_model, H=HYPER_HIDDEN.
        """
        # X = x  [B, N, T]
        n_vars = x.shape[1]                                                     # N

        # Sec. 3.1:  X' = pad_right(X)
        x = self.padding_patch_layer(x)                                         # X'   [B, N, T_pad]

        # Sec. 3.2+3.3:  P~ from _rec_view  (calls _select then _shuffle)
        rec_repr_space      = self._rec_view(x)                                 # P~   [B*N, n, p]
        # Sec. 3.1:       P from _origin_view  (adjacent stride-s patches)
        original_repr_space = self._origin_view(x)                              # P    [B*N, n, p]

        # eq. 12:  E^c = PatchEmbedding^1(P),   E^s = PatchEmbedding^2(P~)
        e_orig = self.value_embedding_org(original_repr_space)                  # E^c   [B*N, n, d]
        e_rec  = self.value_embedding_rec(rec_repr_space)                       # E^s   [B*N, n, d]

        # NEW vs vanilla SRS: produce alpha at runtime from the batch.
        #   ctx       = mean(E^c)               : batch fingerprint
        #   h         = ReLU(W_proj  @ ctx)     : compress to H
        #   alpha_dyn = W_hyper @ h, reshaped   : expand to [n, d]
        #   alpha     = sigmoid(alpha_dyn)      : map to (0, 1)
        ctx       = e_orig.mean(dim=0).mean(dim=0)                              # ctx         [d]
        h         = torch.relu(self.context_proj(ctx))                          # h           [H]
        alpha_dyn = self.hyper(h).view(self.patch_num, e_orig.shape[-1])        # alpha_dyn   [n, d]
        weight    = torch.sigmoid(alpha_dyn)                                    # alpha       [n, d]   (replaces self.alpha)

        # eq. 13:  E~ = alpha * E^c + (1 - alpha) * E^s
        embedding = weight * e_orig + (1.0 - weight) * e_rec                    # E~    [B*N, n, d]

        # eq. 14:  E = E~ + PositionEmbedding(P)
        if self.pos:
            embedding = embedding + self.position_embedding(original_repr_space)  # E    [B*N, n, d]

        # Dropout (not in paper, standard regularization)
        return self.dropout(embedding), n_vars                                  # [B*N, n, d], int


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
# Inserts a standard nn.TransformerEncoder between the SRS module and the
# linear FlattenHead. The encoder treats the n patch embeddings as a sequence
# of length n with d_model channels and runs self-attention across patches.
# Tests the paper's claim (Sec. 4) that "a Linear head is enough".
#
# Two extra hyper-params (read from config via getattr, defaults provided):
#   encoder_n_heads  : number of self-attention heads     (default 8)
#   encoder_n_layers : number of TransformerEncoderLayers (default 2)


class _SRSNetWithEncoderModel(SRSNetModel):
    """SRSNet baseline + Transformer Encoder before the FlattenHead.

    ----------------------------------------------------------------
    1) The problem this extension tries to expose
    ----------------------------------------------------------------
    The SRSNet paper Sec. 4 states that "a simple Linear/MLP Head
    (<= 2 layers)" is enough after the SRS module. This is a strong
    claim: no Transformer, no attention, no extra capacity needed.
    Was that actually true? Or did SRS dominate because no one tried
    adding a real encoder on top?

    ----------------------------------------------------------------
    2) The idea
    ----------------------------------------------------------------
    Plug a standard nn.TransformerEncoder between SRS and the head:

        RevIN(norm) -> SRS -> TransformerEncoder -> FlattenHead -> RevIN(denorm)
                              ^^^^^^^^^^^^^^^^^^^
                              NEW vs vanilla SRSNet

    The encoder sees the n patch embeddings as a sequence of length n
    with d_model channels: each of the n patches attends to all the
    others and gets refined into a context-aware version. If MSE
    improves, the paper's "linear head is enough" claim is incomplete.
    """

    def __init__(self, config):
        """Inherit vanilla SRSNetModel and append a TransformerEncoder.

        Inherited from SRSNetModel via super().__init__():
            self.revin            -- RevIN (norm / denorm)
            self.patch_embedding  -- SRS module
            self.head             -- FlattenHead (Flatten + Linear, eq. 15-16)
        """
        super().__init__(config)
        n_heads  = getattr(config, "encoder_n_heads", 8)
        n_layers = getattr(config, "encoder_n_layers", 2)
        # Standard PyTorch encoder layer: MHA self-attention + FFN + LayerNorm.
        # batch_first=True so the layout matches SRS output [B*N, n, d].
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)    # encoder   [B*N, n, d] -> [B*N, n, d]

    def forward(self, x_enc):
        """Forward: RevIN -> SRS -> Encoder -> FlattenHead -> RevIN^-1.

        Paper notation:
            X       input lookback window               [B, T, N]
            X_norm  RevIN-normalized X                  [B, T, N]
            E       SRS patch embeddings                [B*N, n, d]
            E_attn  E refined by self-attention         [B*N, n, d]   (NEW)
            Y_norm  forecast in normalized scale        [B, L, N]
            Y       forecast in original scale (out)    [B, L, N]
        """
        # X = x_enc  [B, T, N]

        # Sec. 3.1: RevIN normalization (per sample, per channel)
        x_enc = self.revin(x_enc, 'norm')                                       # X_norm   [B, T, N]

        # Move channel axis next to batch (SRS expects [B, N, T])
        x_enc = x_enc.permute(0, 2, 1)                                          # [B, N, T]

        # Sec. 3.2-3.4: SRS module turns the lookback window into n patches embedded in d_model dims
        enc_out, n_vars = self.patch_embedding(x_enc)                           # E   [B*N, n, d]

        # NEW vs vanilla: refine the SRS embeddings with self-attention.
        # The encoder reads E as a "sequence of length n with d channels":
        # each patch attends to every other patch -> patches become
        # context-aware. Shape is unchanged.
        enc_out = self.encoder(enc_out)                                         # E_attn   [B*N, n, d]

        # Restore the channel axis (split B*N back into B and N)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, N, n, d]
        enc_out = enc_out.permute(0, 1, 3, 2)                                   # [B, N, d, n]   FlattenHead wants d before n

        # Sec. 4 eq. 15+16: Y_norm = MLP(Flatten(E_attn))   (one-shot prediction)
        dec_out = self.head(enc_out).permute(0, 2, 1)                           # Y_norm   [B, L, N]

        # Sec. 3.1: RevIN de-normalization (back to the original scale)
        return self.revin(dec_out, 'denorm')                                    # Y   [B, L, N]


class _SelectivityControlsSRSNetEncoderModel(_SelectivityControlsSRSNetModel):
    """Selectivity-controls model + Transformer Encoder (combo).

    Same as ``_SRSNetWithEncoderModel`` except the SRS module is swapped
    for one of our selectivity variants (TASP, HypernetAF, ...) via the
    ``embedding_cls`` attribute inherited from
    ``_SelectivityControlsSRSNetModel``. The Encoder block is identical.

    WHY: ask "does the encoder also help when the SRS itself is modified?"
    -- i.e. the encoder x selectivity factorial.
    """

    def __init__(self, config):
        """Inherit selectivity-controls model + append the Encoder."""
        super().__init__(config)                                                # _SelectivityControlsSRSNetModel sets patch_embedding via embedding_cls
        n_heads  = getattr(config, "encoder_n_heads", 8)
        n_layers = getattr(config, "encoder_n_layers", 2)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)    # encoder   [B*N, n, d] -> [B*N, n, d]

    def forward(self, x_enc):
        """Same pipeline as _SRSNetWithEncoderModel.

        The only thing that differs is the patch_embedding instance:
        SRSTimeAware / SRSHypernetAF / ... instead of vanilla SRS (set
        by the embedding_cls override in subclasses).
        """
        # Sec. 3.1: RevIN norm + channel-first layout
        x_enc = self.revin(x_enc, 'norm')                                       # X_norm   [B, T, N]
        x_enc = x_enc.permute(0, 2, 1)                                          # [B, N, T]

        # SRS variant (TASP / HypernetAF / ...) instead of vanilla SRS
        enc_out, n_vars = self.patch_embedding(x_enc)                           # E   [B*N, n, d]

        # NEW vs vanilla: self-attention over patches
        enc_out = self.encoder(enc_out)                                         # E_attn   [B*N, n, d]

        # Reshape + permute for the head, then linear projection + denorm
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, N, n, d]
        enc_out = enc_out.permute(0, 1, 3, 2)                                   # [B, N, d, n]
        dec_out = self.head(enc_out).permute(0, 2, 1)                           # Y_norm   [B, L, N]
        return self.revin(dec_out, 'denorm')                                    # Y   [B, L, N]


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
