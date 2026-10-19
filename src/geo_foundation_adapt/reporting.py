"""Write the benchmark tables and the label-efficiency figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_foundation_adapt.metrics import format_mean_std

METHOD_LABELS = {
    "linear_probe": "Linear probe",
    "lora": "LoRA",
    "full_finetune": "Full finetune",
}
METHOD_COLORS = {
    "linear_probe": "#0072B2",
    "lora": "#E69F00",
    "full_finetune": "#009E73",
}


def method_label(name: str, lora_rank: int) -> str:
    if name == "lora":
        return f"LoRA (r={lora_rank}, layer4)"
    return METHOD_LABELS.get(name, name)


def write_reports(out_dir: Path, payload: dict, publish_figures: Path | None) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    by_seed = out_dir / "metrics_by_seed.csv"
    seed_header = (
        "seed,label_fraction,labels_per_class,n_train,method,test_accuracy,test_macro_f1,"
        "val_accuracy,val_macro_f1,train_accuracy,train_macro_f1,best_epoch,epochs_ran,"
        "trainable_params,max_epochs,seconds"
    )
    seed_lines = [seed_header]
    for row in payload["runs"]:
        seed_lines.append(
            ",".join(
                [
                    str(row["seed"]),
                    _num(row["label_fraction"]),
                    str(row["labels_per_class"]),
                    str(row["n_train"]),
                    row["method"],
                    _num(row["test_accuracy"]),
                    _num(row["test_macro_f1"]),
                    _num(row["val_accuracy"]),
                    _num(row["val_macro_f1"]),
                    _num(row["train_accuracy"]),
                    _num(row["train_macro_f1"]),
                    str(row["best_epoch"]),
                    str(row["epochs_ran"]),
                    str(row["trainable_params"]),
                    str(row["max_epochs"]),
                    _num(row["seconds"]),
                ]
            )
        )
    by_seed.write_text("\n".join(seed_lines) + "\n", encoding="utf-8")

    summary_path = out_dir / "metrics_table.csv"
    summary_header = (
        "label_fraction,labels_per_class,n_train,method,test_accuracy_mean,test_accuracy_std,"
        "test_macro_f1_mean,test_macro_f1_std,trainable_params,n_seeds"
    )
    summary_lines = [summary_header]
    for row in payload["summary"]:
        summary_lines.append(
            ",".join(
                [
                    _num(row["label_fraction"]),
                    str(row["labels_per_class"]),
                    str(row["n_train"]),
                    row["method"],
                    _num(row["test_accuracy_mean"]),
                    "" if row["test_accuracy_std"] is None else _num(row["test_accuracy_std"]),
                    _num(row["test_macro_f1_mean"]),
                    "" if row["test_macro_f1_std"] is None else _num(row["test_macro_f1_std"]),
                    str(row["trainable_params"]),
                    str(row["n_seeds"]),
                ]
            )
        )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    table_path = out_dir / "table.md"
    table_path.write_text(render_table(payload), encoding="utf-8")

    figure_path = fig_dir / "label_efficiency.png"
    save_figure(payload, figure_path)
    artifacts = {
        "metrics_json": str(metrics_path),
        "metrics_csv": str(summary_path),
        "metrics_by_seed": str(by_seed),
        "table": str(table_path),
        "figure": str(figure_path),
    }
    if publish_figures is not None:
        publish_figures.mkdir(parents=True, exist_ok=True)
        target = publish_figures / figure_path.name
        target.write_bytes(figure_path.read_bytes())
        artifacts["published_figure"] = str(target)
    return artifacts


def render_table(payload: dict) -> str:
    rank = int(payload["adaptation"]["lora_rank"])
    lines = [
        "# EuroSAT label-efficiency results",
        "",
        "Mean ± sample standard deviation across seeds (divisor n−1). "
        "Accuracy and macro-F1 are on the full spatial test band. "
        "The checkpoint is the best validation macro-F1 inside the epoch budget.",
        "",
        "| Labels / class | Fraction of CPU pool | Method | Accuracy | Macro-F1 | Trainable params |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    baseline = payload["baselines"]["spatial_train_majority"]
    lines.append(
        "| — | — | spatial-train majority | "
        f"{baseline['test_accuracy']:.3f} | {baseline['test_macro_f1']:.3f} | 0 |"
    )
    for row in payload["summary"]:
        lines.append(
            "| {k} | {frac:.0%} | {method} | {acc} | {f1} | {params} |".format(
                k=row["labels_per_class"],
                frac=row["label_fraction"],
                method=method_label(row["method"], rank),
                acc=format_mean_std(row["test_accuracy_mean"], row["test_accuracy_std"]),
                f1=format_mean_std(row["test_macro_f1_mean"], row["test_macro_f1_std"]),
                params=row["trainable_params"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def save_figure(payload: dict, path: Path) -> None:
    rank = int(payload["adaptation"]["lora_rank"])
    methods = payload["adaptation"]["methods"]
    summary = payload["summary"]
    fractions = sorted({row["label_fraction"] for row in summary})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharex=True)
    panels = (
        (axes[0], "test_accuracy_mean", "test_accuracy_std", "Test accuracy"),
        (axes[1], "test_macro_f1_mean", "test_macro_f1_std", "Test macro-F1"),
    )
    majority_acc = payload["baselines"]["spatial_train_majority"]["test_accuracy"]
    majority_f1 = payload["baselines"]["spatial_train_majority"]["test_macro_f1"]
    hlines = (majority_acc, majority_f1)
    for ax, mean_key, std_key, title in panels:
        for method in methods:
            rows = [row for row in summary if row["method"] == method]
            rows.sort(key=lambda row: row["label_fraction"])
            xs = [row["labels_per_class"] for row in rows]
            ys = [row[mean_key] for row in rows]
            yerr = [0.0 if row[std_key] is None else row[std_key] for row in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=1.6,
                capsize=3,
                color=METHOD_COLORS.get(method, "#333333"),
                label=method_label(method, rank),
            )
        ax.set_title(title)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    for ax, href in zip(axes, hlines):
        ax.axhline(href, color="#666666", linestyle="--", linewidth=1.0, label="Train majority")
    if len(fractions) > 1:
        for ax in axes:
            ax.set_xscale("log")
    xticks = sorted({row["labels_per_class"] for row in summary})
    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(tick) for tick in xticks])
        ax.set_xlabel("Labeled images per class")
    handles, labels = axes[1].get_legend_handles_labels()
    # The majority line is drawn on both panels; keep a single legend entry.
    dedup = dict(zip(labels, handles))
    axes[1].legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8, loc="lower right")
    backbone = payload["backbone"]["name"]
    n_seeds = payload["protocol"]["n_seeds"]
    fig.suptitle(f"EuroSAT RGB spatial split — {backbone} ({n_seeds} seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _num(value: float) -> str:
    return f"{float(value):.6f}"
