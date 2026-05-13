#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "repro_results"
ENC_IN = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "Weather": 21,
    "Electricity": 321,
    "Solar": 137,
    "Traffic": 862,
}


def _model_command(dataset, model):
    from scripts.repro.paper_repro import _horizon, _read_script_commands, _json_option

    base_model = model
    model_name = None
    adapter = "__KEEP__"
    if model == "SRSPlusPatchTST":
        base_model, model_name, adapter = "PatchTST", "srs_paper.SRSPlusPatchTST", None
    elif model == "SRSPlusCrossformer":
        base_model, model_name, adapter = "Crossformer", "srs_paper.SRSPlusCrossformer", None

    for entry in _read_script_commands(dataset, base_model):
        if "tokens" not in entry:
            continue
        tokens = entry["tokens"]
        if _horizon(tokens) == 720:
            params = _json_option(tokens, "--model-hyper-params")
            params.update({"seq_len": 512, "horizon": 720, "batch_size": 32, "train_drop_last": False})
            return tokens, params, model_name, adapter
    raise SystemExit(f"No H720 official command found for {dataset}/{base_model}")


def _load_wrapper(dataset, model, gpu):
    import torch
    from scripts.repro.paper_repro import _option
    from ts_benchmark.models.model_loader import get_models

    tokens, params, override_model_name, adapter = _model_command(dataset, model)
    model_name = override_model_name or _option(tokens, "--model-name")
    if adapter == "__KEEP__":
        adapter = _option(tokens, "--adapter")
    if adapter == "None":
        adapter = None
    config = {
        "recommend_model_hyper_params": {},
        "models": [
            {
                "adapter": adapter,
                "model_name": model_name,
                "model_hyper_params": params,
            }
        ],
    }
    factory = get_models(config)[0]
    wrapper = factory()
    wrapper.model = wrapper._init_model()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    wrapper.model.to(device)
    return wrapper, params, device


def profile(dataset, model, gpu, iters, warmup):
    import torch

    wrapper, params, device = _load_wrapper(dataset, model, gpu)
    batch = int(params.get("batch_size", 32))
    seq_len = int(params.get("seq_len", 512))
    horizon = int(params.get("horizon", 720))
    label_len = int(params.get("label_len", min(48, seq_len)))
    enc_in = int(params.get("enc_in", ENC_IN[dataset]))

    x = torch.randn(batch, seq_len, enc_in, device=device)
    target = torch.randn(batch, label_len + horizon, enc_in, device=device)
    x_mark = torch.zeros(batch, seq_len, 4, device=device)
    target_mark = torch.zeros(batch, label_len + horizon, 4, device=device)

    optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    def step(train):
        optimizer.zero_grad(set_to_none=True)
        out = wrapper._process(x, target, x_mark, target_mark)["output"]
        loss = criterion(out[:, -horizon:, :], target[:, -horizon:, : out.shape[-1]])
        if train:
            loss.backward()
            optimizer.step()
        return float(loss.detach().cpu())

    for _ in range(warmup):
        step(train=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses = [step(train=True) for _ in range(iters)]
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    train_time = (time.perf_counter() - start) / iters
    train_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            step(train=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    infer_time = (time.perf_counter() - start) / iters

    infer_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    memory_mb = max(train_memory_mb, infer_memory_mb)
    params_count = sum(p.numel() for p in wrapper.model.parameters())
    return {
        "dataset": dataset,
        "model": model,
        "seq_len": seq_len,
        "horizon": horizon,
        "batch_size": batch,
        "parameters": params_count,
        "train_time_s_per_batch": train_time,
        "inference_time_s_per_batch": infer_time,
        "max_gpu_memory_mb": memory_mb,
        "train_gpu_memory_mb": train_memory_mb,
        "inference_gpu_memory_mb": infer_memory_mb,
        "mean_loss": sum(losses) / len(losses),
        "macs": None,
        "note": "MACs are not reported unless a separate supported MAC counter is installed.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--dataset", required=True, choices=["ETTh1", "Solar"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = profile(args.dataset, args.model, int(str(args.gpu).split(",")[0]), args.iters, args.warmup)
    result["table"] = args.table
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(out)


if __name__ == "__main__":
    main()
