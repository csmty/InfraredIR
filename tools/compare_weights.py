#!/usr/bin/env python3
"""Compare two PyTorch checkpoint files tensor by tensor."""

import argparse
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Tuple

import torch


TensorMap = Dict[str, torch.Tensor]


def load_checkpoint(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def flatten_tensors(obj: Any, prefix: str = "") -> TensorMap:
    tensors: TensorMap = {}

    if torch.is_tensor(obj):
        tensors[prefix or "<root>"] = obj.detach().cpu()
        return tensors

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = str(key)
            child_prefix = f"{prefix}.{name}" if prefix else name
            tensors.update(flatten_tensors(value, child_prefix))
        return tensors

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            tensors.update(flatten_tensors(value, child_prefix))
        return tensors

    return tensors


def summarize_non_tensor(obj: Any, prefix: str = "") -> Dict[str, str]:
    values: Dict[str, str] = {}

    if torch.is_tensor(obj):
        return values

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = str(key)
            child_prefix = f"{prefix}.{name}" if prefix else name
            values.update(summarize_non_tensor(value, child_prefix))
        return values

    if isinstance(obj, (list, tuple)):
        if all(not torch.is_tensor(x) and not isinstance(x, (dict, list, tuple)) for x in obj):
            values[prefix or "<root>"] = repr(obj)
        else:
            for idx, value in enumerate(obj):
                child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                values.update(summarize_non_tensor(value, child_prefix))
        return values

    values[prefix or "<root>"] = repr(obj)
    return values


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    af = a.float()
    bf = b.float()
    diff = af - bf
    abs_diff = diff.abs()
    denom = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-12)
    rel = abs_diff / denom

    return {
        "numel": float(a.numel()),
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "rmse": float(torch.sqrt((diff * diff).mean()).item()) if diff.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "same": bool(torch.equal(a, b)),
        "allclose": bool(torch.allclose(af, bf, rtol=1e-5, atol=1e-8)),
    }


def group_name(key: str) -> str:
    parts = key.split(".")
    if key.startswith("task_prompts.") and len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def format_shape(shape: Iterable[int]) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ")"


def print_top_diffs(rows: Iterable[Tuple[str, Dict[str, float]]], topk: int) -> None:
    rows = sorted(rows, key=lambda x: x[1]["max_abs"], reverse=True)
    if not rows:
        print("No comparable tensors with the same shape.")
        return

    print(f"\nTop {min(topk, len(rows))} tensors by max_abs:")
    print("key | numel | mean_abs | max_abs | rmse | mean_rel | max_rel | allclose")
    for key, s in rows[:topk]:
        print(
            f"{key} | {int(s['numel'])} | {s['mean_abs']:.6g} | "
            f"{s['max_abs']:.6g} | {s['rmse']:.6g} | {s['mean_rel']:.6g} | "
            f"{s['max_rel']:.6g} | {s['allclose']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two PyTorch checkpoints tensor by tensor.")
    parser.add_argument("model_a", help="First checkpoint path")
    parser.add_argument("model_b", help="Second checkpoint path")
    parser.add_argument("--topk", type=int, default=30, help="Number of largest-difference tensors to print")
    parser.add_argument("--show-missing", action="store_true", help="Print keys that only exist in one checkpoint")
    parser.add_argument("--show-shape-mismatch", action="store_true", help="Print tensors whose keys match but shapes differ")
    parser.add_argument("--prefix", default="", help="Only compare tensor keys starting with this prefix")
    args = parser.parse_args()

    ckpt_a = load_checkpoint(args.model_a)
    ckpt_b = load_checkpoint(args.model_b)

    tensors_a = flatten_tensors(ckpt_a)
    tensors_b = flatten_tensors(ckpt_b)
    meta_a = summarize_non_tensor(ckpt_a)
    meta_b = summarize_non_tensor(ckpt_b)

    if args.prefix:
        tensors_a = {k: v for k, v in tensors_a.items() if k.startswith(args.prefix)}
        tensors_b = {k: v for k, v in tensors_b.items() if k.startswith(args.prefix)}

    keys_a = set(tensors_a)
    keys_b = set(tensors_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    shape_mismatch = []
    comparable = []
    identical = 0
    allclose = 0
    total_numel = 0
    weighted_abs_sum = 0.0
    max_abs_global = 0.0
    group_stats = defaultdict(lambda: {"count": 0, "numel": 0, "mean_abs_sum": 0.0, "max_abs": 0.0})

    for key in common:
        a = tensors_a[key]
        b = tensors_b[key]
        if tuple(a.shape) != tuple(b.shape):
            shape_mismatch.append((key, tuple(a.shape), tuple(b.shape)))
            continue

        s = tensor_stats(a, b)
        comparable.append((key, s))
        if s["same"]:
            identical += 1
        if s["allclose"]:
            allclose += 1

        numel = int(s["numel"])
        total_numel += numel
        weighted_abs_sum += s["mean_abs"] * numel
        max_abs_global = max(max_abs_global, s["max_abs"])

        g = group_stats[group_name(key)]
        g["count"] += 1
        g["numel"] += numel
        g["mean_abs_sum"] += s["mean_abs"] * numel
        g["max_abs"] = max(g["max_abs"], s["max_abs"])

    print("Checkpoint A:", args.model_a)
    print("Checkpoint B:", args.model_b)
    print(f"Tensor keys: A={len(keys_a)}, B={len(keys_b)}, common={len(common)}")
    print(f"Only in A: {len(only_a)}")
    print(f"Only in B: {len(only_b)}")
    print(f"Shape mismatches: {len(shape_mismatch)}")
    print(f"Comparable tensors: {len(comparable)}")
    print(f"Exactly equal tensors: {identical}")
    print(f"Allclose tensors: {allclose}")

    mean_abs_global = weighted_abs_sum / total_numel if total_numel else math.nan
    print(f"Global tensor numel: {total_numel}")
    print(f"Global mean_abs: {mean_abs_global:.6g}")
    print(f"Global max_abs: {max_abs_global:.6g}")

    meta_changed = sorted(k for k in set(meta_a) & set(meta_b) if meta_a[k] != meta_b[k])
    meta_only_a = sorted(set(meta_a) - set(meta_b))
    meta_only_b = sorted(set(meta_b) - set(meta_a))
    print(f"Non-tensor metadata changed: {len(meta_changed)}")
    print(f"Non-tensor metadata only in A: {len(meta_only_a)}")
    print(f"Non-tensor metadata only in B: {len(meta_only_b)}")

    if meta_changed:
        print("\nChanged non-tensor metadata:")
        for key in meta_changed[:50]:
            print(f"{key}: A={meta_a[key]} | B={meta_b[key]}")

    if group_stats:
        print("\nGroup summary:")
        print("group | tensors | numel | mean_abs | max_abs")
        for group, s in sorted(group_stats.items()):
            mean_abs = s["mean_abs_sum"] / s["numel"] if s["numel"] else 0.0
            print(f"{group} | {s['count']} | {s['numel']} | {mean_abs:.6g} | {s['max_abs']:.6g}")

    print_top_diffs(comparable, args.topk)

    if args.show_shape_mismatch and shape_mismatch:
        print("\nShape mismatches:")
        for key, shape_a, shape_b in shape_mismatch:
            print(f"{key}: A={format_shape(shape_a)} B={format_shape(shape_b)}")

    if args.show_missing:
        if only_a:
            print("\nOnly in A:")
            for key in only_a:
                print(key)
        if only_b:
            print("\nOnly in B:")
            for key in only_b:
                print(key)


if __name__ == "__main__":
    main()
