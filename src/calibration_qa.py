"""Controlled yaw benchmark. Run from the repository root with python -m src.calibration_qa.

Geometry and metrics are implemented locally; supplied I/O and visualization helpers
are reused from starter/. Real labels are held fixed while the calibration changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_frame
from starter.projection import (draw_box2d, overlay_points, perturb_extrinsic,
                                project_velo_to_image, velo_to_cam)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_projection(points, calib, shape):
    uv, depth, mask = project_velo_to_image(points, calib, shape)
    full = np.full((len(points), 2), np.nan)
    full[mask] = uv
    return uv, depth, mask, full


def inside_box(uv, box):
    x1, y1, x2, y2 = box
    return ((uv[:, 0] >= x1) & (uv[:, 0] <= x2)
            & (uv[:, 1] >= y1) & (uv[:, 1] <= y2))


def object_members(points_cam, obj):
    """KITTI bottom-centred, camera-y rotation; local y is in [-h, 0]."""
    h, w, length = obj.dimensions
    if min(h, w, length) <= 0:
        return np.zeros(len(points_cam), dtype=bool)
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (points_cam - obj.location) @ rotation
    tolerance = 1e-6  # Floating-point transforms must not reject exact box corners.
    return (np.isfinite(local).all(axis=1)
            & (np.abs(local[:, 0]) <= length / 2 + tolerance)
            & (local[:, 1] >= -h - tolerance) & (local[:, 1] <= tolerance)
            & (np.abs(local[:, 2]) <= w / 2 + tolerance))


def edge_distance_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    if not edges.any():
        return None
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)


def edge_mean(distance, uv):
    if distance is None or not len(uv):
        return np.nan
    xy = uv.astype(int)
    return float(distance[xy[:, 1], xy[:, 0]].mean())


def annotate(image, title):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (24, 24, 24), -1)
    cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
    return out


def overlay(fr, yaw, depth_range=None):
    calib = perturb_extrinsic(fr["calib"], yaw_deg=yaw)
    uv, depth, _, _ = full_projection(fr["points"], calib, fr["image"].shape)
    if depth_range is not None:
        selected = (depth >= depth_range[0]) & (depth < depth_range[1])
        uv, depth = uv[selected], depth[selected]
    image = overlay_points(fr["image"], uv, depth, radius=1)
    for obj in fr["labels"]:
        image = draw_box2d(image, obj.bbox, label=obj.type)
    return image


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def benchmark(data_root, yaw_levels, object_rows, frame_limit=0):
    frames = list_frames(data_root)
    if frame_limit:
        frames = frames[:frame_limit]
    name = Path(data_root).name
    rows = []
    for i, fid in enumerate(frames):
        fr = load_frame(data_root, fid)
        points, shape = fr["points"], fr["image"].shape
        uv0, _, mask0, full0 = full_projection(points, fr["calib"], shape)
        points_cam = velo_to_cam(points[:, :3], fr["calib"])
        members = [object_members(points_cam, obj) & mask0 for obj in fr["labels"]]
        distance = edge_distance_map(fr["image"])
        for yaw in yaw_levels:
            uv, _, mask, full = full_projection(points, perturb_extrinsic(fr["calib"], yaw_deg=yaw), shape)
            common = mask0 & mask
            displacement = np.linalg.norm(full[common] - full0[common], axis=1)
            any_box = np.zeros(len(uv), dtype=bool)
            matched, denominator = 0, 0
            for object_id, (obj, fixed) in enumerate(zip(fr["labels"], members)):
                any_box |= inside_box(uv, obj.bbox)
                count = int(fixed.sum())
                hits = int((fixed & mask & inside_box(full, obj.bbox)).sum())
                matched += hits
                denominator += count
                object_rows.append(dict(dataset=name, frame_id=fid, yaw_deg=yaw,
                                        object_id=object_id, category=obj.type,
                                        object_depth_m=float(obj.location[2]),
                                        fixed_object_points=count, matched_box_points=hits,
                                        retained_pct=100 * hits / count if count else np.nan))
            # Edge distances use the same visible point identities at both settings.
            edge0 = edge_mean(distance, full0[common])
            edge_now = edge_mean(distance, full[common])
            rows.append(dict(dataset=name, frame_id=fid, yaw_deg=yaw,
                             n_points=len(points), visible_points=int(mask.sum()),
                             fov_pct=100 * mask.mean(),
                             any_box_pct=100 * any_box.mean() if len(any_box) else np.nan,
                             common_points=int(common.sum()),
                             lost_baseline_pct=100 * (mask0 & ~mask).sum() / mask0.sum() if mask0.any() else np.nan,
                             shift_mean_px=float(displacement.mean()) if len(displacement) else np.nan,
                             shift_p95_px=float(np.percentile(displacement, 95)) if len(displacement) else np.nan,
                             object_pairs=denominator, matched_pairs=matched,
                             object_retained_pct=100 * matched / denominator if denominator else np.nan,
                             edge_distance_px=edge_now, edge_baseline_common_px=edge0,
                             edge_delta_px=edge_now - edge0))
        if i % 10 == 0 or i == len(frames) - 1:
            print(f"{name}: {i + 1}/{len(frames)} frames", flush=True)
    return rows


def summarize(table):
    keys = ["dataset", "yaw_deg"]
    metrics = ["fov_pct", "any_box_pct", "lost_baseline_pct", "shift_mean_px", "shift_p95_px",
               "object_retained_pct", "edge_distance_px", "edge_delta_px"]
    summary = table.groupby(keys)[metrics].mean().reset_index()
    counts = table.groupby(keys).size().rename("n_frames").reset_index()
    return summary.merge(counts, on=keys)


def plot_summary(summary, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specifications = [("fov_pct", "Points inside image (%)"),
                      ("shift_mean_px", "Mean pixel shift vs correct calibration"),
                      ("object_retained_pct", "Fixed 3D object points in their 2D box (%)"),
                      ("edge_delta_px", "Change in mean Canny distance (px; lower is better)")]
    for ax, (metric, label) in zip(axes.flat, specifications):
        for dataset, group in summary.groupby("dataset"):
            ax.plot(group.yaw_deg, group[metric], "o-", label=dataset)
        ax.set(xlabel="LiDAR yaw perturbation (degrees)", ylabel=label)
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle("Calibration QA: fixed frames and labels; only LiDAR yaw changes")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select_failure(table):
    baseline = table[table.yaw_deg == 0][["dataset", "frame_id", "fov_pct"]]
    candidates = table[table.yaw_deg.abs() >= 1].merge(baseline, on=["dataset", "frame_id"], suffixes=("", "_baseline"))
    candidates["fov_change_pp"] = candidates.fov_pct - candidates.fov_pct_baseline
    # A weak FOV alarm (<1 percentage point) misses substantial pixel movement.
    weak = candidates[candidates.fov_change_pp.abs() < 1]
    pool = weak if len(weak) else candidates
    return pool.sort_values("shift_mean_px", ascending=False).iloc[0].to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-roots", nargs="+", default=["data/kitti_mini", "data/nuscenes_mini_subset"])
    ap.add_argument("--yaw-deg", nargs="+", type=float, default=[-3, -2, -1, -.5, 0, .5, 1, 2, 3])
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--frame-limit", type=int, default=0, help="0 = every frame; positive values for a quick run")
    args = ap.parse_args()
    if args.frame_limit < 0 or not all(np.isfinite(args.yaw_deg)):
        ap.error("frame limit must be nonnegative and yaw values must be finite")
    yaw = sorted(set(args.yaw_deg + [0.0]))
    if not all(list_frames(root) for root in args.data_roots):
        ap.error("Each dataset must contain at least one frame")
    out = args.out_dir
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, objects = [], []
    for root in args.data_roots:
        rows.extend(benchmark(root, yaw, objects, args.frame_limit))
    write_csv(out / "yaw_perturb_sweep.csv", rows)
    if objects:
        write_csv(out / "yaw_object_metrics.csv", objects)
    table = pd.DataFrame(rows)
    summary = summarize(table)
    summary.to_csv(out / "yaw_summary.csv", index=False, float_format="%.8f")
    plot_summary(summary, figures / "yaw_benchmark.png")
    failure = select_failure(table) if any(abs(v) >= 1 for v in yaw) else None
    if failure:
        root = next(root for root in args.data_roots if Path(root).name == failure["dataset"])
        fr = load_frame(root, failure["frame_id"])
        correct = annotate(overlay(fr, 0), f"{failure['frame_id']} | Correct calibration (yaw 0 deg)")
        drift = annotate(overlay(fr, failure["yaw_deg"]),
                         f"Yaw {failure['yaw_deg']:g} deg | mean shift {failure['shift_mean_px']:.1f}px | FOV change {failure['fov_change_pp']:+.2f}pp")
        save_image(figures / "fail_01_fov_misses_drift.png", np.vstack([correct, drift]))
        failure = {key: value.item() if isinstance(value, np.generic) else value for key, value in failure.items()}
        (out / "failure_case.json").write_text(json.dumps(failure, indent=2, allow_nan=False), encoding="utf-8")
    # Demo depth bands share a KITTI frame, so distance is the sole display filter.
    kitti = next((root for root in args.data_roots if Path(root).name == "kitti_mini"), None)
    if kitti:
        fr = load_frame(kitti, "000011")
        for label, limits in [("near", (0, 15)), ("mid", (15, 30)), ("far", (30, 80))]:
            image = annotate(overlay(fr, 0, limits), f"KITTI 000011 | correct calibration | depth {limits[0]}-{limits[1]}m")
            save_image(figures / f"demo_{label}.png", image)
        save_image(figures / "demo_kitti_yaw_3deg.png", annotate(overlay(fr, 3), "KITTI 000011 | yaw +3 deg"))
    nusc = next((root for root in args.data_roots if Path(root).name == "nuscenes_mini_subset"), None)
    if nusc:
        for scene in ["scene-0103", "scene-1094"]:
            fr = load_frame(nusc, f"{scene}_010")
            save_image(figures / f"demo_{scene}.png", annotate(overlay(fr, 0), f"nuScenes {scene}_010 | correct calibration | ego motion compensated"))
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__, pandas=pd.__version__,
                    data_roots=args.data_roots, yaw_deg=yaw, frame_limit=args.frame_limit,
                    n_frame_settings=len(rows), seed="not applicable: deterministic, no random sampling",
                    metrics="Summary = unweighted mean across frames; object denominator = baseline-visible point-object pairs held fixed",
                    source="KITTI Vision Benchmark Suite; nuScenes (Motional)")
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
