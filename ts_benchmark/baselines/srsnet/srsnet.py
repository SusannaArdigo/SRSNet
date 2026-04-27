from ts_benchmark.baselines.srsnet.models.srsnet_model import SRSNetModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase


MODEL_HYPER_PARAMS = {
    "hidden_size": 128,
    "d_model": 512,
    "freq": "h",
    "patch_len": 24,
    "stride": 24,
    "dropout": 0.2,
    "head_dropout": 0.1,
    "batch_size": 256,
    "lradj": "type1",
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 5,
    "subtract_last": False,
    "affine": True,
    "head_mode": "linear",
    "alpha": 2.0,
    "pos": True
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
        super(SRSNet, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "SRSNet"

    def _init_model(self):
        return SRSNetModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)             
        out_loss = {"output": output}           
        return out_loss
