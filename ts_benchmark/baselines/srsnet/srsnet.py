from ts_benchmark.baselines.srsnet.models.srsnet_model import SRSNetModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase


# --- Default hyper-parameters (paper-faithful) -------------------------------
# Vendor .sh scripts under scripts/multivariate_forecast/<dataset>_script/
# override the dataset-specific values (patch_len, stride, lr, d_model, ...).
# paper_repro.py then injects batch_size=64, train_drop_last=False, plus the
# seed and lambda_aux for PS-SRS variants.
MODEL_HYPER_PARAMS = {
    "hidden_size": 128,          # hidden dim of the MLP scorers inside SRS (scorer_select, scorer_shuffle)
    "d_model": 512,              # embedding dim after PatchEmbedding (d in the paper)
    "freq": "h",                 # data frequency tag (used by some TFB time-encoding utils, unused by SRS)
    "patch_len": 24,             # p in the paper -- patch length
    "stride": 24,                # s in the paper -- patch stride (== patch_len here => non-overlapping)
    "dropout": 0.2,              # dropout inside SRS (final dropout on fused embedding)
    "head_dropout": 0.1,         # dropout on FlattenHead output
    "batch_size": 256,           # default batch (paper_repro.py overrides to 64 paper-faithful)
    "lradj": "type1",            # LR schedule type (decay rule defined in DeepForecastingModelBase)
    "lr": 0.0001,                # initial learning rate
    "num_epochs": 100,           # max training epochs (early stopping usually triggers sooner)
    "num_workers": 0,            # DataLoader workers (paper_repro.py keeps 0 for /dev/shm safety)
    "loss": "MSE",               # paper eq. 16 -- ||Y - Y_hat||^2
    "patience": 5,               # early stopping patience (epochs without val improvement)
    "subtract_last": False,      # RevIN: center on mean (False) vs on the last timestep (True)
    "affine": True,              # RevIN: learnable per-channel scale + bias
    "head_mode": "linear",       # FlattenHead: 'linear' = Linear, anything else = 2-layer MLP w/ SiLU
    "alpha": 2.0,                # SRS adaptive fusion init: sigmoid(2.0) ~= 0.88 (trust original view at start)
    "pos": True,                 # enable sinusoidal positional embedding in SRS
}


class SRSNet(DeepForecastingModelBase):
    """SRSNet adapter for the TFB pipeline.

    Subclasses DeepForecastingModelBase, which handles the training loop,
    learning-rate scheduling, early stopping, save/load, and the rolling
    forecast strategy. We only need to expose three things:

        - model_name  : identifies the model in result CSVs / leaderboard
        - _init_model : creates the underlying nn.Module (SRSNetModel)
        - _process    : single forward pass used by eval/inference

    Training-time forward is provided by the base class via `self.model(...)`.
    """

    def __init__(self, **kwargs):
        """Forward kwargs to the base class, merged with MODEL_HYPER_PARAMS."""
        super(SRSNet, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        """String used by TFB to label this model in result files."""
        return "SRSNet"

    def _init_model(self):
        """Instantiate the underlying nn.Module from the merged config."""
        return SRSNetModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        """Run one forward pass and return the output dict TFB expects.

        input:        [B, T, N] -- past lookback window
        target:       [B, L, N] -- ground-truth future (unused here, the base
                                    class computes the loss outside)
        input_mark / target_mark: time features (unused by SRSNet)
        """
        output = self.model(input)              # [B, L, N] -- forecast in original scale (RevIN-denormalized inside SRSNetModel)
        out_loss = {"output": output}           # TFB expects a dict; loss is computed by the base class
        return out_loss
