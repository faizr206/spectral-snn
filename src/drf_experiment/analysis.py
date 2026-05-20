from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import ensure_dir, load_json


def _family_name(variant: str) -> str:
    if variant == "baseline_drf":
        return "baseline"
    return variant.split("_", 1)[0]


def _extract_epoch_time(manifest: dict[str, Any]) -> float:
    history = manifest.get("history", [])
    if not history:
        return 0.0
    values = [epoch.get("train", {}).get("epoch_time_sec", 0.0) for epoch in history]
    return sum(values) / max(len(values), 1)


def _format_parameter_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _metric_value(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _load_suite_summary_manifests(path: Path) -> list[dict[str, Any]]:
    summary = load_json(path)
    results = summary.get("results", {})
    manifests: list[dict[str, Any]] = []
    for variant in summary.get("variants", []):
        manifest = results.get(variant)
        if manifest is not None:
            manifests.append(manifest)
    if manifests:
        return manifests
    return [manifest for manifest in results.values() if isinstance(manifest, dict)]


def _flatten_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    best = manifest.get("best_test", {})
    cfg = manifest.get("config", {})
    dataset_cfg = cfg.get("dataset", {})
    return {
        "variant": cfg.get("name", "unknown"),
        "family": _family_name(cfg.get("name", "unknown")),
        "dataset": dataset_cfg.get("name", "unknown"),
        "run_dir": manifest.get("run_dir", ""),
        "accuracy": best.get("accuracy", 0.0),
        "loss": best.get("loss", 0.0),
        "spike_rate": best.get("spike_rate", 0.0),
        "energy_mj": best.get("energy_mj", 0.0),
        "effective_energy_proxy_mj": best.get("effective_energy_proxy_mj", best.get("energy_proxy_mj", best.get("energy_mj", 0.0))),
        "train_epoch_time_sec": _extract_epoch_time(manifest),
        "parameter_count": manifest.get("parameter_count", 0),
        "branch_utilization_entropy": best.get("branch_utilization_entropy", 0.0),
        "membrane_amplitude_mean": best.get("membrane_amplitude_mean", 0.0),
        "branch_amplitude_mean": best.get("branch_amplitude_mean", 0.0),
        "gate_mean": best.get("gate_mean", 0.0),
        "gate_entropy": best.get("gate_entropy", 0.0),
        "active_blocks_mean": best.get("active_blocks_mean", 0.0),
        "rho_mean": best.get("rho_mean", 0.0),
        "omega_mean": best.get("omega_mean", 0.0),
    }


def discover_manifests(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    if root.is_file() and root.name == "suite_summary.json":
        return _load_suite_summary_manifests(root)
    suite_summary = root / "suite_summary.json"
    if suite_summary.exists():
        return _load_suite_summary_manifests(suite_summary)
    manifests = []
    for path in sorted(root.rglob("metrics.json")):
        manifests.append(load_json(path))
    return manifests


def load_results_dataframe(root: str | Path) -> pd.DataFrame:
    manifests = discover_manifests(root)
    if not manifests:
        raise FileNotFoundError(f"No metrics.json found under {root}")
    rows = [_flatten_manifest(manifest) for manifest in manifests]
    return pd.DataFrame(rows)


def compute_baseline_deltas(df: pd.DataFrame, baseline_variant: str = "baseline_drf") -> pd.DataFrame:
    records = []
    for dataset, group in df.groupby("dataset"):
        baseline_rows = group[group["variant"] == baseline_variant]
        if baseline_rows.empty:
            enriched = group.copy()
            enriched["delta_accuracy"] = 0.0
            enriched["rel_spike_change"] = 0.0
            enriched["rel_energy_change"] = 0.0
            enriched["rel_time_change"] = 0.0
            enriched["pareto_better"] = False
            enriched["decision_rule"] = "no_baseline"
            records.append(enriched)
            continue
        baseline = baseline_rows.iloc[-1]
        enriched = group.copy()
        enriched["delta_accuracy"] = enriched["accuracy"] - baseline["accuracy"]
        enriched["rel_spike_change"] = (enriched["spike_rate"] - baseline["spike_rate"]) / max(baseline["spike_rate"], 1e-8)
        enriched["rel_energy_change"] = (enriched["energy_mj"] - baseline["energy_mj"]) / max(baseline["energy_mj"], 1e-8)
        enriched["rel_time_change"] = (enriched["train_epoch_time_sec"] - baseline["train_epoch_time_sec"]) / max(
            baseline["train_epoch_time_sec"], 1e-8
        )
        enriched["pareto_better"] = (
            (enriched["accuracy"] >= baseline["accuracy"])
            & (enriched["spike_rate"] <= baseline["spike_rate"])
            & (enriched["train_epoch_time_sec"] <= baseline["train_epoch_time_sec"])
        )
        enriched["decision_rule"] = enriched.apply(_decision_rule, axis=1)
        records.append(enriched)
    return pd.concat(records, ignore_index=True)


def _decision_rule(row: pd.Series) -> str:
    if row["variant"] == "baseline_drf":
        return "baseline"
    if row["delta_accuracy"] >= 0.003 and row["rel_spike_change"] <= 0.10 and row["rel_time_change"] <= 0.10:
        return "accuracy_first"
    if row["delta_accuracy"] >= -0.002 and (row["rel_spike_change"] <= -0.10 or row["rel_energy_change"] <= -0.10):
        return "efficiency_first"
    if row["pareto_better"]:
        return "pareto"
    return "neutral"


def export_summary(root: str | Path, output_dir: str | Path, baseline_variant: str = "baseline_drf") -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    df = compute_baseline_deltas(load_results_dataframe(root), baseline_variant=baseline_variant)
    df.sort_values(["dataset", "accuracy"], ascending=[True, False]).to_csv(output_dir / "summary.csv", index=False)
    return df


def plot_suite_dashboard(root: str | Path, output_dir: str | Path, baseline_variant: str = "baseline_drf") -> list[str]:
    output_dir = ensure_dir(output_dir)
    sns.set_theme(style="whitegrid", context="talk")
    df = export_summary(root, output_dir, baseline_variant=baseline_variant)
    created: list[str] = []

    leaderboard = df.sort_values("accuracy", ascending=False)
    plt.figure(figsize=(12, max(6, len(leaderboard) * 0.35)))
    sns.barplot(data=leaderboard, x="accuracy", y="variant", hue="family", dodge=False)
    plt.title("Variant Accuracy Leaderboard")
    plt.tight_layout()
    path = output_dir / "leaderboard_accuracy.png"
    plt.savefig(path, dpi=220)
    plt.close()
    created.append(str(path))

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="energy_mj",
        y="accuracy",
        hue="family",
        size="parameter_count",
        sizes=(80, 400),
    )
    for _, row in df.iterrows():
        label = f"{row['variant']}\n{_format_parameter_count(row['parameter_count'])}"
        plt.text(row["energy_mj"], row["accuracy"], label, fontsize=8)
    plt.title("Accuracy vs Energy")
    plt.tight_layout()
    path = output_dir / "pareto_accuracy_energy.png"
    plt.savefig(path, dpi=220)
    plt.close()
    created.append(str(path))

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="spike_rate",
        y="accuracy",
        hue="family",
        size="train_epoch_time_sec",
        sizes=(80, 400),
    )
    for _, row in df.iterrows():
        plt.text(row["spike_rate"], row["accuracy"], row["variant"], fontsize=8)
    plt.title("Accuracy vs Spike Rate")
    plt.tight_layout()
    path = output_dir / "pareto_accuracy_spike_rate.png"
    plt.savefig(path, dpi=220)
    plt.close()
    created.append(str(path))

    params = df.sort_values("parameter_count", ascending=False)
    plt.figure(figsize=(12, max(6, len(params) * 0.35)))
    sns.barplot(data=params, x="parameter_count", y="variant", hue="family", dodge=False)
    for index, row in params.reset_index(drop=True).iterrows():
        plt.text(row["parameter_count"], index, _format_parameter_count(row["parameter_count"]), va="center", fontsize=8)
    plt.title("Variant Parameter Count")
    plt.xlabel("Parameters")
    plt.tight_layout()
    path = output_dir / "parameter_count.png"
    plt.savefig(path, dpi=220)
    plt.close()
    created.append(str(path))

    heatmap_df = df.set_index("variant")[["delta_accuracy", "rel_spike_change", "rel_energy_change", "rel_time_change"]]
    plt.figure(figsize=(10, max(5, len(heatmap_df) * 0.35)))
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0.0, annot=True, fmt=".3f")
    plt.title("Improvement Delta Heatmap vs Baseline")
    plt.tight_layout()
    path = output_dir / "delta_heatmap.png"
    plt.savefig(path, dpi=220)
    plt.close()
    created.append(str(path))

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="decision_rule", hue="family")
    plt.title("Decision Rule Outcomes")
    plt.tight_layout()
    path = output_dir / "decision_rules.png"
    plt.savefig(path, dpi=220)
    plt.close()
    created.append(str(path))

    plt.figure(figsize=(10, 6))
    family_mean = df.groupby("family", as_index=False)[["accuracy", "energy_mj", "spike_rate", "parameter_count"]].mean()
    family_long = family_mean.melt(id_vars="family", var_name="metric", value_name="value")
    sns.barplot(data=family_long, x="family", y="value", hue="metric")
    plt.title("Improvement Family Summary")
    plt.tight_layout()
    path = output_dir / "family_summary.png"
    plt.savefig(path, dpi=220)
    plt.close()
    created.append(str(path))

    return created


def plot_training_curves(root: str | Path, output_dir: str | Path, variants: list[str] | None = None) -> str:
    output_dir = ensure_dir(output_dir)
    manifests = discover_manifests(root)
    frames = []
    for manifest in manifests:
        cfg = manifest.get("config", {})
        variant = cfg.get("name", "unknown")
        if variants is not None and variant not in variants:
            continue
        for epoch_entry in manifest.get("history", []):
            train = epoch_entry["train"]
            val = epoch_entry.get("val", {})
            frames.append(
                {
                    "variant": variant,
                    "epoch": epoch_entry["epoch"],
                    "train_accuracy": train.get("accuracy", 0.0),
                    "val_accuracy": val.get("accuracy", 0.0),
                    "train_loss": train.get("loss", 0.0),
                    "val_loss": val.get("loss", 0.0),
                    "spike_rate": train.get("spike_rate", 0.0),
                    "energy_proxy_mj": _metric_value(train, "energy_proxy_mj", "effective_energy_proxy_mj", "energy_mj"),
                }
            )
    if not frames:
        raise FileNotFoundError("No history records found for training curves.")
    df = pd.DataFrame(frames)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.lineplot(data=df, x="epoch", y="train_accuracy", hue="variant", ax=axes[0, 0])
    axes[0, 0].set_title("Train Accuracy")
    sns.lineplot(data=df, x="epoch", y="val_accuracy", hue="variant", ax=axes[0, 1])
    axes[0, 1].set_title("Validation Accuracy")
    sns.lineplot(data=df, x="epoch", y="spike_rate", hue="variant", ax=axes[1, 0])
    axes[1, 0].set_title("Spike Rate")
    energy_df = df.dropna(subset=["energy_proxy_mj"])
    sns.lineplot(data=energy_df, x="epoch", y="energy_proxy_mj", hue="variant", ax=axes[1, 1])
    axes[1, 1].set_title("Energy Proxy")
    for ax in axes.flat:
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=220)
    plt.close()
    return str(path)


def plot_run_diagnostics(run_dir: str | Path, output_dir: str | Path | None = None) -> list[str]:
    run_dir = Path(run_dir)
    manifest = load_json(run_dir / "metrics.json")
    output_dir = ensure_dir(output_dir or (run_dir / "plots"))

    history = manifest.get("history", [])
    if not history:
        raise FileNotFoundError(f"No history found in {run_dir / 'metrics.json'}")
    df = pd.DataFrame(
        [
            {
                "epoch": item["epoch"],
                "train_accuracy": item["train"].get("accuracy", 0.0),
                "val_accuracy": item.get("val", {}).get("accuracy", 0.0),
                "train_loss": item["train"].get("loss", 0.0),
                "val_loss": item.get("val", {}).get("loss", 0.0),
                "spike_rate": item["train"].get("spike_rate", 0.0),
                "energy_proxy_mj": _metric_value(item["train"], "energy_proxy_mj", "effective_energy_proxy_mj", "energy_mj"),
                "branch_entropy": item.get("val", {}).get("branch_utilization_entropy", 0.0),
                "membrane_amplitude_mean": item.get("val", {}).get("membrane_amplitude_mean", 0.0),
            }
            for item in history
        ]
    )

    created: list[str] = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.lineplot(data=df, x="epoch", y="train_accuracy", ax=axes[0, 0], label="train")
    sns.lineplot(data=df, x="epoch", y="val_accuracy", ax=axes[0, 0], label="val")
    axes[0, 0].set_title("Accuracy")
    sns.lineplot(data=df, x="epoch", y="train_loss", ax=axes[0, 1], label="train")
    sns.lineplot(data=df, x="epoch", y="val_loss", ax=axes[0, 1], label="val")
    axes[0, 1].set_title("Loss")
    sns.lineplot(data=df, x="epoch", y="spike_rate", ax=axes[1, 0], label="spike_rate")
    sns.lineplot(data=df.dropna(subset=["energy_proxy_mj"]), x="epoch", y="energy_proxy_mj", ax=axes[1, 0], label="energy_proxy_mj")
    axes[1, 0].set_title("Efficiency")
    sns.lineplot(data=df, x="epoch", y="branch_entropy", ax=axes[1, 1], label="branch_entropy")
    sns.lineplot(data=df, x="epoch", y="membrane_amplitude_mean", ax=axes[1, 1], label="membrane_amplitude")
    axes[1, 1].set_title("Dynamics")
    plt.tight_layout()
    path = output_dir / "run_diagnostics.png"
    plt.savefig(path, dpi=220)
    plt.close()
    created.append(str(path))

    return created
