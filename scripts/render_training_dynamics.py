#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "workspaces" / "v5_4_sam_propagation_memory"
EVENT_FILE = WORKSPACE / "tb_v5_4_sam_propagation_memory" / "events.out.tfevents.1774173384.ldtan.610108.0"
CONFIG_FILE = WORKSPACE / "config.yaml"
OUT_DIR = ROOT / "reports" / "img" / "training_dynamics"


def load_scalars(event_file: Path, wanted: Iterable[str] | None = None) -> Dict[str, List[Tuple[int, float]]]:
    wanted_set = set(wanted) if wanted is not None else None
    series: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for record in RawEventFileLoader(str(event_file)).Load():
        event = event_pb2.Event.FromString(record)
        if not event.summary or not event.summary.value:
            continue
        for value in event.summary.value:
            tag = value.tag
            if wanted_set is not None and tag not in wanted_set:
                continue
            kind = value.WhichOneof("value")
            scalar = None
            if kind == "simple_value":
                scalar = float(value.simple_value)
            elif kind == "tensor":
                arr = tensor_util.make_ndarray(value.tensor)
                try:
                    scalar = float(arr)
                except (TypeError, ValueError):
                    scalar = None
            if scalar is not None:
                series[tag].append((int(event.step), scalar))
    return dict(series)


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) <= 2 or window <= 1:
        return arr
    window = min(window, len(arr))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def to_arrays(points: Sequence[Tuple[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray([p[0] for p in points], dtype=np.float32)
    ys = np.asarray([p[1] for p in points], dtype=np.float32)
    return xs, ys


def normalize_zero_one(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def annotate_phases(ax, boundaries: Sequence[float], labels: Sequence[str], xmax: float) -> None:
    colors = ["#f5efe0", "#eaf4e6", "#e7f0fb", "#f7e7ef"]
    start = 1.0
    for idx, end in enumerate(list(boundaries) + [xmax]):
        ax.axvspan(start, end, color=colors[idx % len(colors)], alpha=0.35, lw=0)
        xmid = 0.5 * (start + end)
        ax.text(xmid, 0.98, labels[idx], fontsize=8, ha="center", va="top", transform=ax.get_xaxis_transform())
        start = end
    for boundary in boundaries:
        ax.axvline(boundary, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.85)


def plot_training_curves(series: Dict[str, List[Tuple[int, float]]], schedule: List[dict], out_path: Path) -> Dict[str, float]:
    epoch_loss_x, epoch_loss_y = to_arrays(series["train/loss_epoch"])
    step_loss_x, step_loss_y = to_arrays(series["train_loss/total"])
    smoothed_step_loss = moving_average(step_loss_y, window=9)

    phase_boundaries = [float(stage["end_epoch"]) for stage in schedule[:-1]]
    phase_labels = [
        "Warm-up",
        "Residual enabled",
        "Stride 1-2",
        "Long-gap stage",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.1), constrained_layout=True)

    ax = axes[0]
    annotate_phases(ax, phase_boundaries, phase_labels, xmax=float(epoch_loss_x[-1]))
    ax.plot(epoch_loss_x, epoch_loss_y, color="#c65d00", linewidth=2.2, marker="o", markersize=3.5)
    ax.set_title("Epoch Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(step_loss_x, step_loss_y, color="#9ca3af", alpha=0.35, linewidth=1.0, label="Logged step loss")
    ax.plot(step_loss_x, smoothed_step_loss, color="#0f766e", linewidth=2.4, label="Smoothed")
    ax.set_title("Step-Level Optimization")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Total loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("AniUnFlow Training Dynamics", fontsize=15, y=1.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "initial_epoch_loss": float(epoch_loss_y[0]),
        "final_epoch_loss": float(epoch_loss_y[-1]),
        "initial_step_loss": float(step_loss_y[0]),
        "final_step_loss": float(step_loss_y[-1]),
    }


def plot_loss_breakdown(series: Dict[str, List[Tuple[int, float]]], schedule: List[dict], out_path: Path) -> None:
    phase_boundaries = [float(stage["end_epoch"]) for stage in schedule[:-1]]
    phase_labels = [
        "Warm-up",
        "Residual enabled",
        "Stride 1-2",
        "Long-gap stage",
    ]
    step_x, _ = to_arrays(series["train_loss/total"])
    max_epoch = float(series["train/loss_epoch"][-1][0])
    max_step = float(step_x[-1])

    def steps_to_epoch(points: Sequence[Tuple[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = to_arrays(points)
        epoch_like = xs / max_step * max_epoch
        return epoch_like, ys

    groups = [
        (
            "Core unsupervised losses",
            [
                ("train_loss/total", "total", "#0f766e", False),
                ("train_loss/photo", "photo", "#c65d00", False),
                ("train_loss/cons", "consistency", "#2563eb", False),
                ("train_loss/smooth", "smoothness", "#6b7280", False),
            ],
        ),
        (
            "Structural and memory losses",
            [
                ("train_loss/v5_segment_warp", "segment warp", "#0f766e", False),
                ("train_loss/v5_sam_memory_consistency", "SAM memory", "#c65d00", False),
                ("train_loss/v5_slot_photo", "slot photo", "#2563eb", False),
            ],
        ),
        (
            "Refinement losses",
            [
                ("train_loss/v5_global_photo", "global photo", "#2563eb", False),
                ("train_loss/v5_piecewise_residual", "piecewise residual", "#dc2626", False),
                ("train_loss/v5_boundary_residual", "boundary residual", "#0f766e", False),
            ],
        ),
        (
            "Activated branch behavior",
            [
                ("train_v5_4/temporal_support_mean", "temporal support", "#2563eb", True),
                ("train_v5_4/residual_mag_mean", "residual magnitude", "#dc2626", True),
                ("train_loss/v5_global_photo", "global photo", "#0f766e", True),
                ("train_loss/v5_sam_memory_consistency", "SAM memory", "#c65d00", True),
            ],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    axes = axes.ravel()

    for ax, (title, items) in zip(axes, groups):
        annotate_phases(ax, phase_boundaries, phase_labels, xmax=max_epoch)
        for tag, label, color, normalize in items:
            xs, ys = steps_to_epoch(series[tag])
            ys_plot = normalize_zero_one(ys) if normalize else ys
            ax.plot(xs, moving_average(ys_plot, window=9), color=color, linewidth=2.0, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch (approx. for step-level logs)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="best")
        if title == "Activated branch behavior":
            ax.set_ylabel("Normalized magnitude")
        else:
            ax.set_ylabel("Loss value")

    fig.suptitle("AniUnFlow Loss Composition and Internal Signals", fontsize=15, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with CONFIG_FILE.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    wanted = {
        "train/loss_epoch",
        "train_loss/total",
        "train_loss/photo",
        "train_loss/cons",
        "train_loss/smooth",
        "train_loss/v5_segment_warp",
        "train_loss/v5_sam_memory_consistency",
        "train_loss/v5_slot_photo",
        "train_loss/v5_dense_slot_consistency",
        "train_loss/v5_global_photo",
        "train_loss/v5_global_dense_consistency",
        "train_loss/v5_piecewise_residual",
        "train_loss/v5_boundary_residual",
        "train_v5_4/temporal_support_mean",
        "train_v5_4/residual_mag_mean",
        "train_v5_4/sam_memory_agreement_mean",
        "train_v5/corr_conf_mean",
    }
    series = load_scalars(EVENT_FILE, wanted)
    schedule = config["runtime"]["schedule"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = plot_training_curves(series, schedule, OUT_DIR / "aniunflow_training_curves.png")
    plot_loss_breakdown(series, schedule, OUT_DIR / "aniunflow_loss_breakdown.png")

    metadata = {
        "workspace": str(WORKSPACE.relative_to(ROOT)),
        "event_file": str(EVENT_FILE.relative_to(ROOT)),
        "schedule": schedule,
        "summary": summary,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
