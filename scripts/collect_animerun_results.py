#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import signal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run this script.") from exc

try:
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto import event_pb2
    from tensorboard.util import tensor_util
except ImportError:
    RawEventFileLoader = None
    event_pb2 = None
    tensor_util = None


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [ROOT / "workspaces", ROOT / "work_dirs", ROOT / "outputs"]
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "animerun_results_table.csv"
REPORT_MD_PATH = REPORT_DIR / "animerun_results_table.md"
TB_TIMEOUT_SEC = 5

METRIC_ORDER = [
    "epe",
    "1px",
    "3px",
    "5px",
    "epe_occ",
    "epe_noc",
    "epe_nonocc",
    "epe_flat",
    "epe_line",
    "epe_s<10",
    "epe_s10-50",
    "epe_s>50",
]

SOURCE_PRIORITY = {
    "metrics_json": 0,
    "metrics_csv": 1,
    "results_json": 2,
    "train_summary": 3,
    "metrics_history": 4,
    "tensorboard": 5,
}


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def flatten_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from flatten_strings(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from flatten_strings(item)
    elif obj is not None:
        yield str(obj)


def load_structured_file(path: Path) -> Dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text()) or {}
    return {}


def config_for_run(run_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    for name in ("config.yaml", "config.json"):
        path = run_dir / name
        if path.exists():
            return path, load_structured_file(path)
    return None, {}


def config_indicates_animerun(cfg: Dict[str, Any]) -> bool:
    haystack = " ".join(s.lower() for s in flatten_strings(cfg))
    return "animerun" in haystack


def path_indicates_animerun(path: Path) -> bool:
    return "animerun" in str(path).lower()


def infer_family(run_dir: Path, cfg: Dict[str, Any]) -> str:
    path_lower = relpath(run_dir).lower()
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    backbone = str(model_cfg.get("backbone", "")).lower() if isinstance(model_cfg, dict) else ""
    if "v5_1_object_memory" in path_lower or "v5_1_object_memory" in backbone:
        return "AniFlowFormerTV5.1"
    if "v5_object_memory" in path_lower:
        return "AniFlowFormerTV5"
    if "upflow" in path_lower:
        return "UPFlow"
    if "unflow" in path_lower and "unsamflow" not in path_lower:
        return "UnFlow"
    if "unsamflow" in path_lower:
        return "UnSAMFlow"
    if "ddflow" in path_lower:
        return "DDFlow"
    if "mdflow" in path_lower:
        return "MDFlow"
    if "segment_aware" in path_lower:
        return "SegmentAware"
    if "global_matching_v3" in path_lower:
        return "AniFlowFormerTV3"
    if "global_matching_v1" in path_lower:
        return "AniFlowFormerT"
    if isinstance(model_cfg, dict):
        if model_cfg.get("backbone"):
            return str(model_cfg["backbone"])
        if model_cfg.get("name"):
            return str(model_cfg["name"])
    return run_dir.name


def infer_research_branch(run_dir: Path, cfg: Dict[str, Any]) -> str:
    path_lower = relpath(run_dir).lower()
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    backbone = str(model_cfg.get("backbone", "")).lower() if isinstance(model_cfg, dict) else ""
    model_name = str(model_cfg.get("name", "")).lower() if isinstance(model_cfg, dict) else ""

    if "v5_1_object_memory" in path_lower or "v5_1_object_memory" in backbone:
        return "V5.1 Object Memory Dense"
    if "v5_object_memory" in path_lower or "v5_object_memory" in backbone or "aniflowformertv5" in model_name:
        return "V5 Object Memory"
    if "v4_6" in path_lower or "v4_5_hybrid_sam" in backbone:
        return "V4 Hybrid SAM"
    if "unsamflow" in path_lower:
        return "UnSAMFlow"
    return "Other"


def normalize_metric_name(name: str) -> str:
    key = name.strip()
    alias = {
        "EPE_all": "epe",
        "epe_nonocc": "epe_nonocc",
        "epe_noc": "epe_noc",
        "epe_nonocc.": "epe_nonocc",
        "epe_epoch": "epe_epoch",
    }
    return alias.get(key, key)


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def metric_payload_from_flat_dict(payload: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        if key in {"dataset", "variant", "timestamp_utc", "num_requested", "checkpoint"}:
            continue
        if isinstance(value, dict):
            continue
        fv = to_float(value)
        if fv is not None:
            metrics[normalize_metric_name(key)] = fv
    return metrics


def make_base_row(run_dir: Path, cfg_path: Optional[Path], cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    workspace = cfg.get("workspace") if isinstance(cfg, dict) else None
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    return {
        "run_dir": relpath(run_dir),
        "run_name": run_dir.name,
        "research_branch": infer_research_branch(run_dir, cfg),
        "model_family": infer_family(run_dir, cfg),
        "workspace_cfg": workspace or "",
        "config_path": relpath(cfg_path) if cfg_path else "",
        "backbone": model_cfg.get("backbone", "") if isinstance(model_cfg, dict) else "",
        "model_name": model_cfg.get("name", "") if isinstance(model_cfg, dict) else "",
        "train_root": data_cfg.get("train_root", "") if isinstance(data_cfg, dict) else "",
        "val_root": data_cfg.get("val_root", "") if isinstance(data_cfg, dict) else "",
    }


def collect_metrics_json(run_dir: Path, base_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("metrics*.json")):
        if path.name in {"metrics_history.jsonl"}:
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            metrics = metric_payload_from_flat_dict(data)
            if metrics:
                row = dict(base_row)
                row.update(metrics)
                row["source_kind"] = "metrics_json"
                row["source_path"] = relpath(path)
                rows.append(row)
    return rows


def collect_metrics_csv(run_dir: Path, base_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("metrics*.csv")):
        metrics: Dict[str, float] = {}
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for item in reader:
                key = normalize_metric_name(item.get("metric", ""))
                value = to_float(item.get("value"))
                if key and value is not None:
                    metrics[key] = value
        if metrics:
            row = dict(base_row)
            row.update(metrics)
            row["source_kind"] = "metrics_csv"
            row["source_path"] = relpath(path)
            rows.append(row)
    return rows


def collect_results_json(run_dir: Path, base_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("results*.json")):
        data = json.loads(path.read_text())
        experiments = data.get("experiments")
        if not isinstance(experiments, list):
            continue
        for exp in experiments:
            metrics = metric_payload_from_flat_dict(exp.get("metrics", {}))
            if not metrics:
                continue
            row = dict(base_row)
            row.update(metrics)
            row["run_name"] = exp.get("name", row["run_name"])
            row["source_kind"] = "results_json"
            row["source_path"] = relpath(path)
            row["variant"] = data.get("variant", "")
            row["checkpoint"] = exp.get("checkpoint", "")
            rows.append(row)
    return rows


def collect_train_summary(run_dir: Path, base_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = run_dir / "train_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    row = dict(base_row)
    best_epe = to_float(data.get("best_epe"))
    final_epoch = to_float(data.get("final_epoch"))
    final_step = to_float(data.get("final_step"))
    if best_epe is not None:
        row["epe"] = best_epe
    if final_epoch is not None:
        row["selected_epoch"] = int(final_epoch)
    if final_step is not None:
        row["selected_step"] = int(final_step)
    row["source_kind"] = "train_summary"
    row["source_path"] = relpath(path)
    return [row]


def collect_metrics_history(run_dir: Path, base_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = run_dir / "metrics_history.jsonl"
    if not path.exists():
        return []
    best_row: Optional[Dict[str, Any]] = None
    best_epe = float("inf")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        val_epe = to_float(item.get("val_epe"))
        if val_epe is None:
            continue
        if val_epe < best_epe:
            best_epe = val_epe
            best_row = item
    if best_row is None:
        return []
    row = dict(base_row)
    row["epe"] = best_epe
    if "epoch" in best_row:
        row["selected_epoch"] = int(best_row["epoch"])
    if "step" in best_row:
        row["selected_step"] = int(best_row["step"])
    row["source_kind"] = "metrics_history"
    row["source_path"] = relpath(path)
    return [row]


def find_tb_dir(run_dir: Path) -> Optional[Path]:
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.startswith("tb"):
            return child
    return None


def latest_event_files(tb_dir: Path, limit: int = 3) -> List[Path]:
    files = [path for path in tb_dir.iterdir() if path.is_file() and path.name.startswith("events.out.tfevents.")]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def collect_tensorboard(run_dir: Path, base_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if RawEventFileLoader is None or event_pb2 is None or tensor_util is None:
        return []
    tb_dir = find_tb_dir(run_dir)
    if tb_dir is None:
        return []
    for event_file in latest_event_files(tb_dir):
        metrics_by_step: Dict[int, Dict[str, float]] = {}
        timed_out = False

        def _timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError

        try:
            previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TB_TIMEOUT_SEC)
            loader = RawEventFileLoader(str(event_file))
            for record in loader.Load():
                event = event_pb2.Event.FromString(record)
                if not event.summary or not event.summary.value:
                    continue
                step = int(event.step)
                for value in event.summary.value:
                    tag = value.tag
                    if not tag.startswith("val/") or tag.endswith("_epoch"):
                        continue
                    value_kind = value.WhichOneof("value")
                    scalar_value: Optional[float] = None
                    if value_kind == "simple_value":
                        scalar_value = float(value.simple_value)
                    elif value_kind == "tensor":
                        arr = tensor_util.make_ndarray(value.tensor)
                        try:
                            scalar_value = float(arr)
                        except (TypeError, ValueError):
                            scalar_value = None
                    if scalar_value is None:
                        continue
                    metric_name = normalize_metric_name(tag.split("/", 1)[1])
                    metrics_by_step.setdefault(step, {})[metric_name] = scalar_value
        except TimeoutError:
            timed_out = True
        except Exception:
            continue
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

        if not metrics_by_step:
            continue

        if any("epe" in step_metrics for step_metrics in metrics_by_step.values()):
            best_step, best_metrics = min(
                (
                    (step, step_metrics)
                    for step, step_metrics in metrics_by_step.items()
                    if "epe" in step_metrics
                ),
                key=lambda item: item[1]["epe"],
            )
        else:
            best_step, best_metrics = sorted(metrics_by_step.items())[0]

        row = dict(base_row)
        row["source_kind"] = "tensorboard_partial" if timed_out else "tensorboard"
        row["source_path"] = relpath(event_file)
        row["selected_step"] = int(best_step)
        row["selected_epoch"] = int(best_step)
        row.update(best_metrics)
        return [row]
    return []


def candidate_run_dirs() -> List[Path]:
    runs: set[Path] = set()
    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if name.startswith("events.out.tfevents."):
                runs.add(path.parent.parent)
            elif name in {
                "config.yaml",
                "config.json",
                "train_summary.json",
                "metrics_history.jsonl",
            }:
                runs.add(path.parent)
            elif name.startswith("metrics") and path.suffix in {".json", ".csv"}:
                runs.add(path.parent)
            elif name.startswith("results") and path.suffix == ".json":
                runs.add(path.parent)
    return sorted(runs)


def pick_best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def key(row: Dict[str, Any]) -> Tuple[int, float, str]:
        priority = SOURCE_PRIORITY.get(row.get("source_kind", ""), 99)
        epe = row.get("epe")
        epe_sort = float(epe) if isinstance(epe, (int, float)) else float("inf")
        return (priority, epe_sort, row.get("source_path", ""))

    return min(rows, key=key)


def collect_rows_for_run(run_dir: Path) -> List[Dict[str, Any]]:
    cfg_path, cfg = config_for_run(run_dir)
    if not (config_indicates_animerun(cfg) or path_indicates_animerun(run_dir)):
        return []

    base_row = make_base_row(run_dir, cfg_path, cfg)
    rows: List[Dict[str, Any]] = []
    rows.extend(collect_metrics_json(run_dir, base_row))
    rows.extend(collect_metrics_csv(run_dir, base_row))
    rows.extend(collect_results_json(run_dir, base_row))
    rows.extend(collect_train_summary(run_dir, base_row))
    rows.extend(collect_metrics_history(run_dir, base_row))

    if not rows:
        rows.extend(collect_tensorboard(run_dir, base_row))

    if not rows:
        missing_row = dict(base_row)
        missing_row["source_kind"] = "missing_metrics"
        missing_row["source_path"] = ""
        missing_row["num_sources_found"] = 0
        return [missing_row]

    best = pick_best_row(rows)
    best["num_sources_found"] = len(rows)
    return [best]


def final_columns(rows: List[Dict[str, Any]]) -> List[str]:
    leading = [
        "run_name",
        "research_branch",
        "model_family",
        "run_dir",
        "source_kind",
        "source_path",
        "config_path",
        "workspace_cfg",
        "backbone",
        "model_name",
        "selected_epoch",
        "selected_step",
        "num_sources_found",
    ]
    trailing = ["train_root", "val_root", "variant", "checkpoint"]
    extra_metrics = [metric for metric in METRIC_ORDER if any(metric in row for row in rows)]
    seen = set(leading + extra_metrics + trailing)
    extras = sorted(
        key for row in rows for key in row.keys() if key not in seen
    )
    return leading + extra_metrics + trailing + extras


def write_markdown(rows: List[Dict[str, Any]]) -> None:
    columns = ["run_name", "research_branch", "model_family", "source_kind", "epe", "1px", "3px", "5px", "source_path"]
    lines = [
        "# AnimeRun Results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows: List[Dict[str, Any]] = []
    candidates = candidate_run_dirs()
    for index, run_dir in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {relpath(run_dir)}")
        rows.extend(collect_rows_for_run(run_dir))

    rows.sort(
        key=lambda row: (
            row.get("model_family", ""),
            float(row.get("epe", float("inf"))) if row.get("epe") is not None else float("inf"),
            row.get("run_name", ""),
        )
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    columns = final_columns(rows)
    with REPORT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_markdown(rows)

    print(f"Wrote {len(rows)} rows to {REPORT_PATH}")


if __name__ == "__main__":
    main()
