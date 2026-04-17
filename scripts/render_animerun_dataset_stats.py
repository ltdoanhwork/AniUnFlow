#!/usr/bin/env python3
"""Render dataset/statistics figures for the AnimeRun thesis section."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import frame_utils


SPLITS = ("train", "test")
PRIMARY_MODALITIES = (
    "Frame_Anime",
    "Flow",
    "Segment",
    "SegMatching",
    "UnmatchedForward",
    "UnmatchedBackward",
    "contour",
    "LineArea",
)
EXTRA_MODALITIES = (
    "sam_masks",
    "SAM_Masks",
    "SAM_Masks_v2",
)
COLORS = {"train": "#1f77b4", "test": "#e67e22"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "AnimeRun_v2",
        help="Path to AnimeRun_v2 root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "img",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--flow-stride",
        type=int,
        default=8,
        help="Spatial subsampling stride used for motion statistics.",
    )
    parser.add_argument(
        "--max-flows-per-scene",
        type=int,
        default=8,
        help="Maximum number of forward-flow files sampled from each scene.",
    )
    return parser.parse_args()


def infer_family(scene_name: str) -> str:
    if scene_name.startswith("agent"):
        return "agent"
    if scene_name.startswith("cami"):
        return "cami"
    if scene_name.startswith("sprite"):
        return "sprite"
    return "other"


def count_original_frames(scene_dir: Path) -> int:
    original_dir = scene_dir / "original"
    if not original_dir.exists():
        return 0
    return len(sorted(original_dir.glob("*.png")))


def count_forward_flows(scene_dir: Path) -> int:
    forward_dir = scene_dir / "forward"
    if not forward_dir.exists():
        return 0
    return len(sorted(forward_dir.glob("*.flo")))


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def load_flow(path: Path) -> np.ndarray:
    return frame_utils.readFlow(str(path)).astype(np.float32)


def sample_flow_paths(forward_dir: Path, max_flows_per_scene: int) -> list[Path]:
    paths = sorted(forward_dir.glob("*.flo"))
    if max_flows_per_scene <= 0 or len(paths) <= max_flows_per_scene:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=max_flows_per_scene, dtype=int)
    return [paths[idx] for idx in indices]


def collect_split_summary(
    dataset_root: Path,
    flow_stride: int,
    max_flows_per_scene: int,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    split_summary: dict[str, dict[str, object]] = {}
    pair_stats: list[dict[str, object]] = []

    for split in SPLITS:
        frame_root = dataset_root / split / "Frame_Anime"
        flow_root = dataset_root / split / "Flow"
        scene_names = sorted(p.name for p in frame_root.iterdir() if p.is_dir())
        frame_counts = []
        flow_counts = []
        family_counts = Counter()
        scene_motion_p95 = defaultdict(list)

        for scene_name in scene_names:
            frame_count = count_original_frames(frame_root / scene_name)
            flow_count = count_forward_flows(flow_root / scene_name)
            frame_counts.append(frame_count)
            flow_counts.append(flow_count)
            family_counts[infer_family(scene_name)] += 1

            forward_dir = flow_root / scene_name / "forward"
            for flow_path in sample_flow_paths(forward_dir, max_flows_per_scene):
                flow = load_flow(flow_path)
                flow = flow[::flow_stride, ::flow_stride]
                valid = np.isfinite(flow).all(axis=-1)
                if not np.any(valid):
                    continue
                magnitude = np.linalg.norm(flow[valid], axis=-1)
                mean_mag = float(np.mean(magnitude))
                p95_mag = safe_percentile(magnitude, 95)
                max_mag = float(np.max(magnitude))
                pair_stats.append(
                    {
                        "split": split,
                        "scene": scene_name,
                        "frame": flow_path.stem,
                        "mean_mag": mean_mag,
                        "p95_mag": p95_mag,
                        "max_mag": max_mag,
                    }
                )
                scene_motion_p95[scene_name].append(p95_mag)

        modality_coverage = []
        for modality in PRIMARY_MODALITIES:
            modality_root = dataset_root / split / modality
            available = set(p.name for p in modality_root.iterdir() if p.is_dir()) if modality_root.exists() else set()
            coverage = len(available & set(scene_names)) / max(len(scene_names), 1)
            modality_coverage.append(coverage)

        extra_coverage = []
        for modality in EXTRA_MODALITIES:
            modality_root = dataset_root / modality / split / "Frame_Anime"
            available = set(p.name for p in modality_root.iterdir() if p.is_dir()) if modality_root.exists() else set()
            extra_coverage.append(len(available & set(scene_names)) / max(len(scene_names), 1))

        split_summary[split] = {
            "scene_names": scene_names,
            "scene_count": len(scene_names),
            "frame_counts": frame_counts,
            "pair_counts": flow_counts,
            "total_frames": int(sum(frame_counts)),
            "total_pairs": int(sum(flow_counts)),
            "family_counts": family_counts,
            "modality_coverage": modality_coverage,
            "extra_coverage": extra_coverage,
            "mean_clip_length": safe_mean(frame_counts),
            "mean_pair_length": safe_mean(flow_counts),
            "scene_motion_p95": {scene: safe_mean(values) for scene, values in scene_motion_p95.items()},
        }

    return split_summary, pair_stats


def render_dataset_overview(split_summary: dict[str, dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.patch.set_facecolor("white")

    splits = list(SPLITS)
    x = np.arange(len(splits))
    width = 0.24
    scene_counts = [split_summary[split]["scene_count"] for split in splits]
    total_frames = [split_summary[split]["total_frames"] for split in splits]
    total_pairs = [split_summary[split]["total_pairs"] for split in splits]

    ax = axes[0, 0]
    ax.bar(x - width, scene_counts, width, label="Scenes", color="#34495e")
    ax.bar(x, total_frames, width, label="Frames", color="#4ea8de")
    ax.bar(x + width, total_pairs, width, label="Flow pairs", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels([split.title() for split in splits])
    ax.set_title("Split-level scale")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    family_order = ["agent", "cami", "sprite", "other"]
    bottom = np.zeros(len(splits), dtype=float)
    family_colors = {
        "agent": "#355070",
        "cami": "#6d597a",
        "sprite": "#b56576",
        "other": "#e56b6f",
    }
    for family in family_order:
        values = [split_summary[split]["family_counts"].get(family, 0) for split in splits]
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=family_colors[family],
            width=0.55,
            label=family.title(),
        )
        bottom += np.asarray(values, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([split.title() for split in splits])
    ax.set_title("Scene-family composition")
    ax.set_ylabel("Scenes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)

    ax = axes[1, 0]
    bins = np.arange(0, max(max(split_summary[s]["frame_counts"]) for s in splits) + 15, 10)
    for split in splits:
        ax.hist(
            split_summary[split]["frame_counts"],
            bins=bins,
            alpha=0.72,
            color=COLORS[split],
            label=f"{split.title()} clips",
            edgecolor="white",
        )
    ax.set_title("Clip-length distribution")
    ax.set_xlabel("Frames per scene")
    ax.set_ylabel("Number of scenes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    heatmap = np.array(
        [
            split_summary["train"]["modality_coverage"] + split_summary["train"]["extra_coverage"],
            split_summary["test"]["modality_coverage"] + split_summary["test"]["extra_coverage"],
        ]
    )
    im = ax.imshow(heatmap, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    labels = list(PRIMARY_MODALITIES) + list(EXTRA_MODALITIES)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Train", "Test"])
    ax.set_title("Scene-level modality coverage")
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            value = heatmap[row, col]
            ax.text(
                col,
                row,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.5 else "#223",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coverage")

    fig.suptitle("AnimeRun overview for AniUnFlow", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_motion_statistics(
    split_summary: dict[str, dict[str, object]],
    pair_stats: list[dict[str, object]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.patch.set_facecolor("white")

    pair_stats_by_split = {
        split: [item for item in pair_stats if item["split"] == split]
        for split in SPLITS
    }

    ax = axes[0, 0]
    max_mean = max(item["mean_mag"] for item in pair_stats) if pair_stats else 1.0
    bins = np.linspace(0, math.ceil(max_mean / 2.0) * 2.0, 28)
    for split in SPLITS:
        ax.hist(
            [item["mean_mag"] for item in pair_stats_by_split[split]],
            bins=bins,
            alpha=0.68,
            label=f"{split.title()} pairs",
            color=COLORS[split],
            edgecolor="white",
        )
    ax.set_title("Average flow magnitude per pair")
    ax.set_xlabel("Mean motion magnitude (pixels)")
    ax.set_ylabel("Number of frame pairs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    max_p95 = max(item["p95_mag"] for item in pair_stats) if pair_stats else 1.0
    bins = np.linspace(0, math.ceil(max_p95 / 5.0) * 5.0, 26)
    for split in SPLITS:
        ax.hist(
            [item["p95_mag"] for item in pair_stats_by_split[split]],
            bins=bins,
            alpha=0.65,
            label=f"{split.title()} tail motion",
            color=COLORS[split],
            edgecolor="white",
        )
    ax.set_title("High-motion tail (95th percentile)")
    ax.set_xlabel("95th-percentile magnitude (pixels)")
    ax.set_ylabel("Number of frame pairs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for split in SPLITS:
        values = pair_stats_by_split[split]
        ax.scatter(
            [item["mean_mag"] for item in values],
            [item["max_mag"] for item in values],
            s=18,
            alpha=0.5,
            color=COLORS[split],
            label=split.title(),
        )
    ax.set_title("Pair difficulty spread")
    ax.set_xlabel("Mean motion magnitude (pixels)")
    ax.set_ylabel("Maximum motion magnitude (pixels)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    scene_scores: dict[str, float] = {}
    for split in SPLITS:
        for scene, score in split_summary[split]["scene_motion_p95"].items():
            scene_scores[f"{split}:{scene}"] = score
    hardest = sorted(scene_scores.items(), key=lambda item: item[1], reverse=True)[:8]
    labels = [name.split(":", 1)[1] for name, _ in hardest]
    values = [score for _, score in hardest]
    colors = [COLORS[name.split(":", 1)[0]] for name, _ in hardest]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Scenes with strongest motion tail")
    ax.set_xlabel("Average 95th-percentile magnitude")
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Motion-statistics profile of AnimeRun", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    split_summary, pair_stats = collect_split_summary(
        args.dataset_root,
        max(args.flow_stride, 1),
        args.max_flows_per_scene,
    )

    overview_path = args.output_dir / "animerun_dataset_overview.png"
    motion_path = args.output_dir / "animerun_motion_statistics.png"
    render_dataset_overview(split_summary, overview_path)
    render_motion_statistics(split_summary, pair_stats, motion_path)

    print(f"[DatasetStats] Saved {overview_path}")
    print(f"[DatasetStats] Saved {motion_path}")


if __name__ == "__main__":
    main()
