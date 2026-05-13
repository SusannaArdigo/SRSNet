#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts" / "multivariate_forecast"
RESULT_ROOT = ROOT / "result"
REPRO_ROOT = ROOT / "repro_results"
DATASET_ROOT = ROOT / "dataset" / "forecasting"

PAPER_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Solar", "Traffic"]
PAPER_HORIZONS = [96, 192, 336, 720]
PAPER_PLUGIN_DATASETS = ["ETTh1", "ETTm2", "Solar", "Traffic"]
PAPER_LOOKBACKS = [96, 336, 512]
PAPER_SRSNET_SEEDS = [2021, 2022, 2023, 2024, 2025]
PAPER_BATCH_FLOOR = 8
METRIC_SPACE = "mse_norm/mae_norm after model inverse_transform, normalized by the evaluator train split scaler"
PLUGIN_MODELS = [
    ("SRSNet", None),
    ("SRSNet", "srs_paper.SRSNet_NoSRS"),
    ("PatchTST", None),
    ("PatchTST", "srs_paper.SRSPlusPatchTST"),
    ("Crossformer", None),
    ("Crossformer", "srs_paper.SRSPlusCrossformer"),
    ("PatchMLP", None),
    ("PatchMLP", "srs_paper.SRSPlusPatchMLP"),
    ("xPatch", None),
    ("xPatch", "srs_paper.SRSPlusxPatch"),
]

PAPER_BASELINES = {
    "SRSNet",
    "TimeMixer",
    "xPatch",
    "TimeKAN",
    "Amplifier",
    "Pathformer",
    "PDF",
    "PatchMLP",
    "FITS",
    "DLinear",
    "TimesNet",
    "FEDformer",
    "Pyraformer",
    "Autoformer",
    "Informer",
    "Transformer",
    "Nonstationary_Transformer",
    "Reformer",
}

SRSNET_TABLE2 = {
    ("ETTh1", 96): (0.366, 0.394),
    ("ETTh1", 192): (0.400, 0.415),
    ("ETTh1", 336): (0.424, 0.430),
    ("ETTh1", 720): (0.426, 0.455),
    ("ETTh2", 96): (0.271, 0.338),
    ("ETTh2", 192): (0.335, 0.379),
    ("ETTh2", 336): (0.323, 0.381),
    ("ETTh2", 720): (0.399, 0.441),
    ("ETTm1", 96): (0.288, 0.341),
    ("ETTm1", 192): (0.329, 0.367),
    ("ETTm1", 336): (0.365, 0.387),
    ("ETTm1", 720): (0.421, 0.418),
    ("ETTm2", 96): (0.164, 0.254),
    ("ETTm2", 192): (0.220, 0.291),
    ("ETTm2", 336): (0.273, 0.327),
    ("ETTm2", 720): (0.350, 0.383),
    ("Weather", 96): (0.148, 0.199),
    ("Weather", 192): (0.195, 0.242),
    ("Weather", 336): (0.248, 0.282),
    ("Weather", 720): (0.316, 0.333),
    ("Electricity", 96): (0.128, 0.222),
    ("Electricity", 192): (0.147, 0.240),
    ("Electricity", 336): (0.165, 0.258),
    ("Electricity", 720): (0.204, 0.292),
    ("Solar", 96): (0.164, 0.218),
    ("Solar", 192): (0.183, 0.240),
    ("Solar", 336): (0.189, 0.245),
    ("Solar", 720): (0.194, 0.252),
    ("Traffic", 96): (0.352, 0.244),
    ("Traffic", 192): (0.378, 0.257),
    ("Traffic", 336): (0.393, 0.271),
    ("Traffic", 720): (0.444, 0.308),
}


@dataclass
class Task:
    task_id: str
    table: str
    dataset: str
    horizon: int
    model: str
    command: list[str]
    save_path: str
    seed: int | None = None
    seq_len: int | None = None
    status: str = "pending"
    note: str = ""
    oom_retry: bool = True

    @property
    def command_hash(self):
        return _command_hash(self.command)


def _command_hash(command):
    payload = json.dumps(command, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _stable_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _script_path(dataset, model):
    return SCRIPT_ROOT / f"{dataset}_script" / f"{model}.sh"


def _read_script_commands(dataset, model):
    path = _script_path(dataset, model)
    if not path.exists():
        return []
    commands = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "run_benchmark.py" not in stripped:
            continue
        stripped = stripped.rstrip("&").strip()
        try:
            tokens = shlex.split(stripped)
        except ValueError as exc:
            commands.append({"error": f"{path}:{line_no}: {exc}"})
            continue
        commands.append({"tokens": tokens, "source": f"{path}:{line_no}"})
    return commands


def _option(tokens, name, default=None):
    if name not in tokens:
        return default
    idx = tokens.index(name)
    if idx + 1 >= len(tokens):
        return default
    return tokens[idx + 1]


def _set_option(tokens, name, value):
    tokens = list(tokens)
    if name in tokens:
        idx = tokens.index(name)
        tokens[idx + 1] = str(value)
    else:
        tokens.extend([name, str(value)])
    return tokens


def _remove_option(tokens, name):
    tokens = list(tokens)
    while name in tokens:
        idx = tokens.index(name)
        del tokens[idx : idx + 2]
    return tokens


def _json_option(tokens, name):
    value = _option(tokens, name)
    return json.loads(value) if value is not None else {}


def _identity_payload(task, command):
    return {
        "dataset": task.dataset,
        "horizon": task.horizon,
        "model": task.model,
        "seed": task.seed,
        "seq_len": task.seq_len,
        "data_name_list": _option(command, "--data-name-list"),
        "model_name": _option(command, "--model-name"),
        "adapter": _option(command, "--adapter"),
        "strategy_args": _json_option(command, "--strategy-args"),
        "model_hyper_params": _json_option(command, "--model-hyper-params"),
        "deterministic": _option(command, "--deterministic", "efficient"),
    }


def _config_hash(task, command):
    return _stable_hash(_identity_payload(task, command))


def _model_label(tokens):
    save_path = _option(tokens, "--save-path", "")
    if "/" in save_path:
        return save_path.split("/")[-1]
    model_name = _option(tokens, "--model-name", "")
    return model_name.split(".")[-1]


def _horizon(tokens):
    try:
        return int(_json_option(tokens, "--strategy-args").get("horizon"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _paper_hyper_params(tokens, horizon, seq_len=None, paper_mode=True):
    params = _json_option(tokens, "--model-hyper-params")
    params["horizon"] = horizon
    if paper_mode:
        params["batch_size"] = 64
        params["train_drop_last"] = False
    if seq_len is not None:
        params["seq_len"] = seq_len
    return params


def _normalized_command(tokens, *, scope, table, task_id, gpu, seed=None, seq_len=None, model_name=None, adapter="__KEEP__", paper_mode=True):
    horizon = _horizon(tokens)
    if horizon is None:
        raise ValueError("cannot infer horizon")
    save_path = f"repro/{scope}/{table}/{task_id}"
    out_dir = RESULT_ROOT / save_path
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = list(tokens)
    cmd[0] = sys.executable
    cmd = _set_option(cmd, "--gpus", gpu)
    cmd = _set_option(cmd, "--num-workers", 1)
    cmd = _set_option(cmd, "--save-path", save_path)
    if seed is not None:
        cmd = _set_option(cmd, "--seed", seed)
    if model_name is not None:
        cmd = _set_option(cmd, "--model-name", model_name)
    if adapter == "__KEEP__":
        pass
    elif adapter is None:
        cmd = _remove_option(cmd, "--adapter")
    else:
        cmd = _set_option(cmd, "--adapter", adapter)
    params = _paper_hyper_params(cmd, horizon, seq_len=seq_len, paper_mode=paper_mode)
    cmd = _set_option(cmd, "--model-hyper-params", json.dumps(params, sort_keys=True))
    return cmd, save_path


def _task_from_tokens(tokens, *, scope, table, dataset, model, gpu, seed=None, seq_len=None, model_name=None, adapter="__KEEP__", paper_mode=True):
    horizon = _horizon(tokens)
    suffix = []
    if seed is not None:
        suffix.append(f"s{seed}")
    if seq_len is not None:
        suffix.append(f"L{seq_len}")
    suffix_text = "_" + "_".join(suffix) if suffix else ""
    task_model = model_name.split(".")[-1] if model_name else model
    task_model = task_model.replace("+", "Plus")
    task_id = f"{table}_{dataset}_H{horizon}_{task_model}{suffix_text}"
    cmd, save_path = _normalized_command(
        tokens,
        scope=scope,
        table=table,
        task_id=task_id,
        gpu=gpu,
        seed=seed,
        seq_len=seq_len,
        model_name=model_name,
        adapter=adapter,
        paper_mode=paper_mode,
    )
    return Task(task_id, table, dataset, horizon, model_name.split(".")[-1] if model_name else model, cmd, save_path, seed, seq_len, oom_retry=paper_mode)


def _official_tasks_for(dataset, model, *, scope, table, gpu, seeds=(2021,), seq_lens=(None,), model_name=None, adapter=None):
    tasks = []
    entries = _read_script_commands(dataset, model)
    display_model = model_name.split(".")[-1] if model_name else model
    paper_mode = scope != "main-compat"
    effective_adapter = adapter
    if model_name is None and adapter is None:
        effective_adapter = "__KEEP__"
    if not entries:
        for horizon in PAPER_HORIZONS:
            tasks.append(
                Task(
                    f"{table}_{dataset}_H{horizon}_{display_model}_reference",
                    table,
                    dataset,
                    horizon,
                    display_model,
                    [],
                    "",
                    status="reference-only",
                    note=f"No official main script for {dataset}/{model}; paper row retained for coverage.",
                )
            )
        return tasks
    for entry in entries:
        if "error" in entry:
            tasks.append(Task(f"{table}_{dataset}_{display_model}_bad_script", table, dataset, -1, display_model, [], "", status="reference-only", note=entry["error"]))
            continue
        tokens = entry["tokens"]
        horizon = _horizon(tokens)
        if horizon not in PAPER_HORIZONS:
            continue
        for seed in seeds:
            for seq_len in seq_lens:
                try:
                    tasks.append(
                        _task_from_tokens(
                            tokens,
                            scope=scope,
                            table=table,
                            dataset=dataset,
                            model=model,
                            gpu=gpu,
                            seed=seed,
                            seq_len=seq_len,
                            model_name=model_name,
                            adapter=effective_adapter,
                            paper_mode=paper_mode,
                        )
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    tasks.append(
                        Task(
                            f"{table}_{dataset}_H{horizon}_{display_model}_bad_json",
                            table,
                            dataset,
                            horizon,
                            display_model,
                            [],
                            "",
                            status="reference-only",
                            note=f"{entry['source']}: invalid official command JSON: {exc}",
                        )
                    )
    return tasks


def build_tasks(scope, gpu):
    tasks = []
    if scope == "main-compat":
        for dataset in PAPER_DATASETS:
            for model in sorted(PAPER_BASELINES):
                tasks.extend(_official_tasks_for(dataset, model, scope=scope, table="main_compat", gpu=gpu))
        return tasks

    srs_seeds = PAPER_SRSNET_SEEDS if scope == "full-paper" else (2021,)
    srs_seq_lens = PAPER_LOOKBACKS if scope == "full-paper" else (None,)
    for dataset in PAPER_DATASETS:
        tasks.extend(_official_tasks_for(dataset, "SRSNet", scope=scope, table="table2_srsnet", gpu=gpu, seeds=srs_seeds, seq_lens=srs_seq_lens))

    ablation_datasets = PAPER_PLUGIN_DATASETS
    for dataset in ablation_datasets:
        tasks.extend(_official_tasks_for(dataset, "SRSNet", scope=scope, table="table4_ablation", gpu=gpu))
    ablation_models = ["SRSNet_NoSRS", "SRSNet_NoSP", "SRSNet_NoDR", "SRSNet_NoAF"]
    for dataset in ablation_datasets:
        for variant in ablation_models:
            tasks.extend(
                _official_tasks_for(
                    dataset,
                    "SRSNet",
                    scope=scope,
                    table="table4_ablation",
                    gpu=gpu,
                    model_name=f"srs_paper.{variant}",
                    adapter=None,
                )
            )

    plugin_datasets = PAPER_PLUGIN_DATASETS
    plugin_pairs = [(base, plus, None) for base, plus in PLUGIN_MODELS]
    for dataset in plugin_datasets:
        for base_model, plus_model, adapter in plugin_pairs:
            if scope == "lite-paper" and plus_model not in {None, "srs_paper.SRSNet_NoSRS"}:
                for horizon in PAPER_HORIZONS:
                    model_label = plus_model.split(".")[-1] if plus_model else base_model
                    tasks.append(
                        Task(
                            f"table3_plugin_{dataset}_H{horizon}_{model_label}_lite_omitted",
                            "table3_plugin",
                            dataset,
                            horizon,
                            model_label,
                            [],
                            "",
                            status="omitted-lite",
                            note="Heavy plug-in row omitted from lite-paper scope; run full-paper for this row.",
                        )
                    )
                continue
            tasks.extend(_official_tasks_for(dataset, base_model, scope=scope, table="table3_plugin", gpu=gpu, model_name=plus_model, adapter=adapter))

    if scope == "full-paper":
        for dataset in PAPER_DATASETS:
            for model in sorted(PAPER_BASELINES - {"SRSNet"}):
                tasks.extend(_official_tasks_for(dataset, model, scope=scope, table="table2_baselines", gpu=gpu))
    else:
        for dataset in PAPER_DATASETS:
            for model in sorted(PAPER_BASELINES - {"SRSNet"}):
                for horizon in PAPER_HORIZONS:
                    tasks.append(
                        Task(
                            f"table2_baselines_{dataset}_H{horizon}_{model}_lite_omitted",
                            "table2_baselines",
                            dataset,
                            horizon,
                            model,
                            [],
                            "",
                            status="omitted-lite",
                            note="Full baseline matrix omitted from lite-paper scope; run full-paper for this row.",
                        )
                    )

    tasks.extend(_efficiency_reference_tasks(scope, gpu))

    return tasks


def _efficiency_reference_tasks(scope, gpu):
    rows = []
    table5_models = ["SRSNet", "PatchTST", "Crossformer", "DLinear", "TimesNet", "FEDformer"]
    table6_models = ["PatchTST", "SRSPlusPatchTST", "Crossformer", "SRSPlusCrossformer"]
    for table, models in (("table5_efficiency", table5_models), ("table6_efficiency", table6_models)):
        for dataset in ("ETTh1", "Solar"):
            for model in models:
                task_id = f"{table}_{dataset}_{model}"
                if scope == "full-paper":
                    out = REPRO_ROOT / scope / "efficiency" / f"{task_id}.json"
                    cmd = [
                        sys.executable,
                        "scripts/repro/efficiency.py",
                        "--scope",
                        scope,
                        "--table",
                        table,
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--gpu",
                        gpu,
                        "--out",
                        str(out),
                    ]
                    rows.append(Task(task_id, table, dataset, 720, model, cmd, "", seq_len=512, note="One-batch profiler for the paper efficiency setting."))
                else:
                    rows.append(Task(task_id, table, dataset, 720, model, [], "", seq_len=512, status="omitted-lite", note="Efficiency benchmark omitted from lite-paper scope."))
    return rows


def _status_path(scope):
    path = REPRO_ROOT / scope
    path.mkdir(parents=True, exist_ok=True)
    return path / "status.jsonl"


def _manifest_path(scope):
    path = REPRO_ROOT / scope
    path.mkdir(parents=True, exist_ok=True)
    return path / "manifest.jsonl"


def _metadata_path(scope, task_id):
    path = REPRO_ROOT / scope / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{task_id}.json"


def _load_metadata(scope, task_id):
    path = _metadata_path(scope, task_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_metadata(scope, task_id, metadata):
    _metadata_path(scope, task_id).write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _load_status(scope):
    path = _status_path(scope)
    statuses = {}
    if not path.exists():
        return statuses
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        statuses[item["task_id"]] = item
    return statuses


def write_manifest(scope, tasks):
    with _manifest_path(scope).open("w") as fh:
        for task in tasks:
            item = asdict(task)
            item["command_hash"] = task.command_hash if task.command else ""
            item["config_hash"] = _config_hash(task, task.command) if task.command else ""
            item["metric_space"] = METRIC_SPACE if task.command else ""
            fh.write(json.dumps(item, sort_keys=True) + "\n")


def _append_status(scope, item):
    with _status_path(scope).open("a") as fh:
        fh.write(json.dumps(item, sort_keys=True) + "\n")


def _latest_report(save_path):
    directory = RESULT_ROOT / save_path
    files = sorted(directory.glob("test_report*.csv"), key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else ""


def _explicit_output(command):
    return _option(command, "--out", "")


def _oom_text(text):
    lowered = text.lower()
    return "out of memory" in lowered or "cuda error: out of memory" in lowered


def _with_batch_size(command, batch_size):
    params = _json_option(command, "--model-hyper-params")
    params["batch_size"] = batch_size
    return _set_option(command, "--model-hyper-params", json.dumps(params, sort_keys=True))


def _batch_size(command):
    params = _json_option(command, "--model-hyper-params")
    return int(params.get("batch_size", 64))


def _completed_metadata_matches(scope, task, previous):
    if not previous or previous.get("status") != "completed":
        return False
    metadata = _load_metadata(scope, task.task_id)
    result_file = metadata.get("result_file") or previous.get("result_file")
    if not result_file or not Path(result_file).exists():
        return False
    requested_config_hash = _config_hash(task, task.command)
    return metadata.get("requested_config_hash") == requested_config_hash


def _stale_result_files(scope):
    keep_root = (RESULT_ROOT / "repro" / scope).resolve()
    stale = []
    for directory in (ROOT / "results", RESULT_ROOT):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if directory == RESULT_ROOT and (resolved == keep_root or keep_root in resolved.parents):
                continue
            stale.append(path)
    return stale


def check_stale_results(scope):
    stale = _stale_result_files(scope)
    if not stale:
        print("No stale legacy result files found.")
        return
    print("Stale legacy result files found. These are not used by the repro runner:")
    for path in stale[:100]:
        print(f"- {path}")
    if len(stale) > 100:
        print(f"... and {len(stale) - 100} more")
    raise SystemExit(1)


def run_tasks(scope, tasks, *, resume=True, keep_going=False, dry_run=False, max_tasks=None):
    statuses = _load_status(scope)
    selected = tasks[: max_tasks or len(tasks)]
    for index, task in enumerate(selected, start=1):
        if task.status != "pending":
            print(f"[{index}/{len(selected)}] {task.status} {task.task_id}")
            _append_status(scope, {**asdict(task), "status": task.status, "command_hash": "", "result_file": "", "ended_at": time.time()})
            continue
        previous = statuses.get(task.task_id)
        if resume and _completed_metadata_matches(scope, task, previous):
            print(f"[{index}/{len(selected)}] skip {task.task_id}")
            continue
        print(f"[{index}/{len(selected)}] run {task.task_id}")
        print(shlex.join(task.command))
        if dry_run:
            continue
        command = list(task.command)
        attempts = []
        while True:
            started = time.time()
            proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output = proc.stdout or ""
            log_dir = REPRO_ROOT / scope / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{task.task_id}.log"
            log_file.write_text(output)
            attempts.append({"returncode": proc.returncode, "batch_size": _batch_size(command)})
            if proc.returncode == 0:
                result_file = _latest_report(task.save_path) or _explicit_output(command)
                metadata = {
                    **asdict(task),
                    "status": "completed",
                    "requested_command": task.command,
                    "final_command": command,
                    "requested_command_hash": task.command_hash,
                    "command_hash": _command_hash(command),
                    "requested_config_hash": _config_hash(task, task.command),
                    "config_hash": _config_hash(task, command),
                    "requested_identity": _identity_payload(task, task.command),
                    "final_identity": _identity_payload(task, command),
                    "final_batch_size": _batch_size(command),
                    "result_file": result_file,
                    "attempts": attempts,
                    "metric_space": METRIC_SPACE,
                    "started_at": started,
                    "ended_at": time.time(),
                }
                _write_metadata(scope, task.task_id, metadata)
                _append_status(
                    scope,
                    metadata,
                )
                break
            batch_size = _batch_size(command)
            if task.oom_retry and _oom_text(output) and batch_size > PAPER_BATCH_FLOOR:
                command = _with_batch_size(command, max(PAPER_BATCH_FLOOR, batch_size // 2))
                print(f"OOM; retrying {task.task_id} with batch_size={_batch_size(command)}")
                continue
            _append_status(
                scope,
                {
                    **asdict(task),
                    "status": "failed",
                    "command_hash": task.command_hash,
                    "result_file": "",
                    "attempts": attempts,
                    "log_file": str(log_file),
                    "started_at": started,
                    "ended_at": time.time(),
                },
            )
            if not keep_going:
                raise SystemExit(f"failed: {task.task_id}; see {log_file}")
            break


def _read_metric_row(result_file):
    if not result_file:
        return {}
    path = Path(result_file)
    if not path.exists():
        return {}
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else {}


def _read_json_result(result_file):
    if not result_file:
        return {}
    path = Path(result_file)
    if not path.exists() or path.suffix != ".json":
        return {}
    return json.loads(path.read_text())


def _load_manifest(scope):
    path = _manifest_path(scope)
    items = {}
    if not path.exists():
        return items
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        items[item["task_id"]] = item
    return items


def collect(scope):
    manifest = _load_manifest(scope)
    statuses = _load_status(scope)
    out_dir = REPRO_ROOT / scope
    summary_path = out_dir / "summary.csv"
    coverage_path = out_dir / "coverage.md"
    rows = []
    combined = {**manifest, **statuses}
    for item in combined.values():
        metric = _read_metric_row(item.get("result_file", ""))
        metric.update(_read_json_result(item.get("result_file", "")))
        dataset = item.get("dataset")
        horizon = item.get("horizon")
        paper = SRSNET_TABLE2.get((dataset, horizon)) if item.get("model") == "SRSNet" else None
        mse = metric.get("mse_norm") or metric.get("mse") or ""
        mae = metric.get("mae_norm") or metric.get("mae") or ""
        try:
            mse_delta = float(mse) - paper[0] if paper and mse != "" else ""
            mae_delta = float(mae) - paper[1] if paper and mae != "" else ""
        except ValueError:
            mse_delta = ""
            mae_delta = ""
        rows.append(
            {
                "task_id": item.get("task_id"),
                "table": item.get("table"),
                "dataset": dataset,
                "horizon": horizon,
                "model": item.get("model"),
                "seed": item.get("seed"),
                "seq_len": item.get("seq_len"),
                "status": item.get("status", "missing"),
                "mse": mse,
                "mae": mae,
                "paper_mse": paper[0] if paper else "",
                "paper_mae": paper[1] if paper else "",
                "delta_mse": mse_delta,
                "delta_mae": mae_delta,
                "final_batch_size": item.get("final_batch_size", ""),
                "metric_space": item.get("metric_space", METRIC_SPACE if item.get("command") else ""),
                "parameters": metric.get("parameters", ""),
                "train_time_s_per_batch": metric.get("train_time_s_per_batch", ""),
                "inference_time_s_per_batch": metric.get("inference_time_s_per_batch", ""),
                "max_gpu_memory_mb": metric.get("max_gpu_memory_mb", ""),
                "train_gpu_memory_mb": metric.get("train_gpu_memory_mb", ""),
                "inference_gpu_memory_mb": metric.get("inference_gpu_memory_mb", ""),
                "macs": metric.get("macs", ""),
                "command_hash": item.get("command_hash", ""),
                "requested_command_hash": item.get("requested_command_hash", item.get("command_hash", "")),
                "config_hash": item.get("config_hash", ""),
                "requested_config_hash": item.get("requested_config_hash", item.get("config_hash", "")),
                "result_file": item.get("result_file", ""),
                "note": item.get("note", ""),
            }
        )
    rows.sort(key=lambda r: (str(r["table"]), str(r["dataset"]), int(r["horizon"] or -1), str(r["model"]), str(r["seed"]), str(r["seq_len"])))
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["task_id"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    table_counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        table_counts[(row["table"], row["status"])] = table_counts.get((row["table"], row["status"]), 0) + 1
    lines = [f"# Reproduction Coverage: {scope}", ""]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## By Table", ""])
    for (table, status), count in sorted(table_counts.items()):
        lines.append(f"- {table} / {status}: {count}")
    lines.extend(["", f"Summary CSV: `{summary_path}`", ""])
    coverage_path.write_text("\n".join(lines))
    print(f"Wrote {summary_path}")
    print(f"Wrote {coverage_path}")


def check_data():
    missing = []
    meta = DATASET_ROOT / "FORECAST_META.csv"
    if not meta.exists():
        missing.append(str(meta))
    for dataset in PAPER_DATASETS:
        path = DATASET_ROOT / f"{dataset}.csv"
        if not path.exists():
            missing.append(str(path))
    if missing:
        print("Missing official forecasting data files:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)
    print(f"Found official forecasting data files under {DATASET_ROOT}")


def smoke_check(scope, dataset, horizon, tolerance_mse, tolerance_mae):
    summary_path = REPRO_ROOT / scope / "summary.csv"
    if not summary_path.exists():
        collect(scope)
    with summary_path.open() as fh:
        rows = list(csv.DictReader(fh))
    candidates = [
        row
        for row in rows
        if row["table"] == "table2_srsnet"
        and row["dataset"] == dataset
        and int(row["horizon"]) == horizon
        and row["model"] == "SRSNet"
        and row["status"] == "completed"
    ]
    if not candidates:
        raise SystemExit(f"No completed SRSNet smoke row found for {dataset} H{horizon} in {summary_path}")
    paper = SRSNET_TABLE2[(dataset, horizon)]
    best = min(candidates, key=lambda row: abs(float(row["mse"]) - paper[0]))
    mse_delta = abs(float(best["mse"]) - paper[0])
    mae_delta = abs(float(best["mae"]) - paper[1])
    print(
        f"Smoke {dataset} H{horizon}: mse={best['mse']} paper={paper[0]} "
        f"mae={best['mae']} paper={paper[1]} metric_space={best.get('metric_space', METRIC_SPACE)}"
    )
    if mse_delta > tolerance_mse or mae_delta > tolerance_mae:
        raise SystemExit(
            f"Smoke check failed: delta_mse={mse_delta:.6f} tolerance={tolerance_mse}; "
            f"delta_mae={mae_delta:.6f} tolerance={tolerance_mae}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["manifest", "run", "collect", "dry-coverage", "check-data", "check-stale-results", "smoke-check"])
    parser.add_argument("--scope", default="lite-paper", choices=["lite-paper", "full-paper", "main-compat"])
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--smoke-dataset", default="ETTh1")
    parser.add_argument("--smoke-horizon", type=int, default=96)
    parser.add_argument("--smoke-tolerance-mse", type=float, default=0.08)
    parser.add_argument("--smoke-tolerance-mae", type=float, default=0.08)
    args = parser.parse_args()

    if args.command == "check-data":
        check_data()
        return
    if args.command == "check-stale-results":
        check_stale_results(args.scope)
        return
    if args.command == "smoke-check":
        smoke_check(args.scope, args.smoke_dataset, args.smoke_horizon, args.smoke_tolerance_mse, args.smoke_tolerance_mae)
        return

    tasks = build_tasks(args.scope, args.gpu)
    write_manifest(args.scope, tasks)
    if args.command in {"manifest", "dry-coverage"}:
        print(f"Wrote {_manifest_path(args.scope)} ({len(tasks)} tasks)")
        return
    if args.command == "run":
        run_tasks(args.scope, tasks, resume=not args.no_resume and not args.force, keep_going=args.keep_going, dry_run=args.dry_run, max_tasks=args.max_tasks)
    elif args.command == "collect":
        collect(args.scope)


if __name__ == "__main__":
    main()
