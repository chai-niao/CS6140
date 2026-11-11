"""Figures and comparison tables for the report and the slides.

    python plots.py --runs "Model A (XLM-R)=runs/xlmr_full" "Model B (Qwen2.5-0.5B)=runs/qwen_full" \
                    --sweep "0=runs/sweep_lam0" "0.5=runs/sweep_lam05" "1=runs/sweep_lam1" "2=runs/sweep_lam2"

Writes to --out_dir (default results/):
    convergence_<run>.png   training loss of both heads and the dev F1 over optimizer steps  (one figure per run)
    f1_by_pair.png          Anticipatory F1 of every pair: naive baseline of Update 1 vs the trained models
    f1_by_pair.md           the same numbers as a table
    comparison.md           headline numbers of every run side by side
    lambda_sweep.png / .md  mean F1, sigma and duration accuracy as a function of lambda_duration
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import setup_path  # noqa: E402, F401
from common import markdown_table  # noqa: E402

TEAL = "#00897B"      # first model / switch head (same hue family as the slides)
ORANGE = "#E38B2C"    # second model / duration head
GRAY = "#B8B8B8"      # naive baseline (reference, hatched)
INK = "#222222"
MUTED = "#6B6B6B"
SERIES_COLORS = [TEAL, ORANGE, "#5B6472", "#8E5BB5"]  # fixed order, never cycled
DEFAULT_BASELINE = setup_path.UPDATE1_DIR / "results" / "baseline_f1.csv"
PAIR_LABEL = {"Arabic_eng": "Arabic", "Can_Eng": "Cantonese", "Chinese_eng": "Mandarin", "French_eng": "French",
              "German_Eng": "German", "German_French": "German-French", "Hindi_eng": "Hindi", "Italian_eng": "Italian",
              "Japanese_eng": "Japanese", "Korean_eng": "Korean", "Malay_eng": "Malay", "Philippines_eng": "Filipino",
              "Russian_eng": "Russian", "Spanish_eng": "Spanish", "Vietnamese_eng": "Vietnamese"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
})


def load_run(path: Path) -> dict:
    run = {"path": path, "summary": json.loads((path / "summary.json").read_text(encoding="utf-8"))}
    run["test"] = pd.read_csv(path / "test_metrics.csv")
    run["train_log"] = pd.read_csv(path / "train_log.csv") if (path / "train_log.csv").exists() else None
    run["dev_log"] = pd.read_csv(path / "dev_log.csv") if (path / "dev_log.csv").exists() else None
    return run


def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def smooth(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=max(span, 1), adjust=False).mean()


def plot_convergence(name: str, run: dict, out: Path) -> None:
    """One run: training loss of both heads, dev loss of both heads, dev scores; x axis in epochs."""
    log, dev = run["train_log"], run["dev_log"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"wspace": 0.3})
    span = max(len(log) // 40, 1)
    for col, color, label in (("loss_switch", TEAL, "switch head"), ("loss_duration", ORANGE, "duration head")):
        ax1.plot(log["epoch"], log[col], color=color, alpha=0.18, linewidth=0.8)
        ax1.plot(log["epoch"], smooth(log[col], span), color=color, linewidth=2, label=label)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy")
    ax1.set_title(f"{name}: training loss", loc="left", color=INK)
    ax1.legend(loc="upper right")
    if dev is not None and len(dev):
        for col, color, marker, label in (("dev_loss_switch", TEAL, "o", "switch head"), ("dev_loss_duration", ORANGE, "s", "duration head")):
            if col in dev:
                ax2.plot(dev["epoch"], dev[col], color=color, marker=marker, markersize=5, linewidth=2, label=label)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("cross-entropy")
        ax2.set_title("dev loss (100 conversations per pair)", loc="left", color=INK)
        ax2.legend(loc="center right")
        f1col = "dev_mean_F1_tuned" if "dev_mean_F1_tuned" in dev else "dev_mean_F1_at_0.5"
        ax3.plot(dev["epoch"], dev[f1col], color=TEAL, marker="o", markersize=5, linewidth=2, label="mean F1, thresholds tuned")
        if "dev_mean_dur_acc" in dev:
            ax3.plot(dev["epoch"], dev["dev_mean_dur_acc"], color=ORANGE, marker="s", markersize=5, linewidth=2, label="duration accuracy at switches")
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("dev score")
        ax3.set_title("dev scores", loc="left", color=INK)
        ax3.legend(loc="center right")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_compare(runs: dict[str, dict], out: Path) -> None:
    """All runs in one figure: one panel per head; smoothed training loss (line) and dev loss (markers), x in epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"wspace": 0.25})
    for ax, (tcol, dcol, head) in zip(axes, (("loss_switch", "dev_loss_switch", "switch head"),
                                             ("loss_duration", "dev_loss_duration", "duration head"))):
        for j, (name, run) in enumerate(runs.items()):
            color = SERIES_COLORS[j]
            log, dev = run["train_log"], run["dev_log"]
            if log is not None:
                ax.plot(log["epoch"], smooth(log[tcol], max(len(log) // 40, 1)), color=color, linewidth=2, label=f"{name}, training")
            if dev is not None and dcol in dev:
                ax.plot(dev["epoch"], dev[dcol], color=color, marker="o", markersize=5, linewidth=0, label=f"{name}, dev")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax.set_title(head, loc="left", color=INK)
    axes[1].legend(loc="upper right", fontsize=8.5)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_f1_by_pair(runs: dict[str, dict], baseline: pd.DataFrame | None, out_png: Path, out_md: Path) -> None:
    table = pd.DataFrame({"pair": sorted(next(iter(runs.values()))["test"]["pair"])}).set_index("pair")
    if baseline is not None:
        table["naive baseline"] = baseline.set_index("pair")["F1_token"]
    for name, run in runs.items():
        table[name] = run["test"].set_index("pair")["F1"]
    table = table.sort_values(table.columns[0])
    n_series = len(table.columns)
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(11, 4))
    x = range(len(table))
    for j, col in enumerate(table.columns):
        offset = (j - (n_series - 1) / 2) * width
        if col == "naive baseline":
            kw = {"color": GRAY, "hatch": "////", "edgecolor": "white", "linewidth": 0.5}
        else:
            kw = {"color": SERIES_COLORS[list(runs).index(col)], "edgecolor": "white", "linewidth": 0.5}
        mean, sigma = table[col].mean(), table[col].std(ddof=0)
        ax.bar([i + offset for i in x], table[col], width=width * 0.92, label=f"{col}  (mean {mean:.3f}, sigma {sigma:.3f})", **kw)
    ax.set_xticks(list(x))
    ax.set_xticklabels([PAIR_LABEL.get(p, p) for p in table.index], rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Anticipatory F1 (test split)")
    ax.set_title("Anticipatory F1 per language pair", loc="left", color=INK, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n_series, borderaxespad=0.2)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    rows = [[pair] + [f"{v:.3f}" for v in vals] for pair, vals in table.iterrows()]
    rows.append(["**mean**"] + [f"**{table[c].mean():.3f}**" for c in table.columns])
    rows.append(["**sigma**"] + [f"**{table[c].std(ddof=0):.3f}**" for c in table.columns])
    out_md.write_text("# Anticipatory F1 per pair (test split)\n\n" + markdown_table(["pair"] + list(table.columns), rows) + "\n", encoding="utf-8")


def comparison_table(runs: dict[str, dict], baseline: pd.DataFrame | None, out_md: Path) -> None:
    header = ["run", "backbone", "trainable params", "lambda_sw / lambda_dur", "steps", "train min", "mean F1", "sigma", "F1 at 0.5",
              "dur acc", "macro F1 by type", "F1 tag", "F1 inter", "F1 intra"]
    rows = []
    if baseline is not None:
        f1 = baseline["F1_token"]
        rows.append(["naive baseline (Update 1)", "P(switch \\| token)", "", "", "", "", f"{f1.mean():.3f}", f"{f1.std(ddof=0):.3f}", "",
                     f"{baseline['dur_acc_token'].mean():.3f}", "", "", "", ""])
    for name, run in runs.items():
        s, t = run["summary"], run["summary"]["test"]
        config = json.loads((run["path"] / "config.json").read_text(encoding="utf-8"))
        rows.append([name, s["backbone"], f"{config['trainable_parameters'] / 1e6:.0f}M", f"{s['lambda_switch']:g} / {s['lambda_duration']:g}",
                     s["steps"], s["train_minutes"], f"{t['mean_F1']:.3f}", f"{t['sigma_F1']:.3f}", f"{t['mean_F1_at_0.5']:.3f}",
                     f"{t['mean_dur_acc']:.3f}", f"{t['mean_F1_macro_by_type']:.3f}",
                     f"{t.get('mean_F1_tag', float('nan')):.3f}", f"{t.get('mean_F1_inter', float('nan')):.3f}", f"{t.get('mean_F1_intra', float('nan')):.3f}"])
    out_md.write_text("# Runs side by side (test split, all pairs)\n\n" + markdown_table(header, rows) + "\n", encoding="utf-8")


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def plot_sweep(sweep: dict[str, dict], out_png: Path, out_md: Path) -> None:
    """Entries named by a number are lambda_duration values (plotted); other names are extra runs (table only),
    e.g. "no class weights" or "duration only". A head trained with weight 0 gets n/a for its metric."""
    numeric = sorted(((float(k), r) for k, r in sweep.items() if _is_number(k)), key=lambda kv: kv[0])
    if numeric:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"wspace": 0.35})
        panels = (("mean_F1", "mean Anticipatory F1", 0.0), ("sigma_F1", "sigma of F1 across pairs", 0.0),
                  ("mean_dur_acc", "duration accuracy at switches", 1e-9))
        for ax, (key, label, min_lam) in zip(axes, panels):
            pts = [(lam, r["summary"]["test"][key]) for lam, r in numeric if lam >= min_lam]
            ax.plot([x for x, _ in pts], [y for _, y in pts], color=TEAL, marker="o", markersize=6, linewidth=2)
            ax.set_xlabel("lambda_duration  (lambda_switch = 1)")
            ax.set_title(label, loc="left", color=INK)
            ax.set_xticks([lam for lam, _ in numeric])
            ys = [y for _, y in pts]
            mid, half = (max(ys) + min(ys)) / 2, max((max(ys) - min(ys)) * 0.65, 0.025)
            ax.set_ylim(mid - half, mid + half)  # at least 0.05 tall, so tiny differences do not look large
            ax.set_axisbelow(True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
    header = ["run", "lambda_switch", "lambda_duration", "class weights", "mean F1", "sigma", "F1 at 0.5", "dur acc",
              "macro F1 by type", "final switch loss", "final duration loss"]
    ordered = [(f"{lam:g}", r) for lam, r in numeric] + [(k, r) for k, r in sweep.items() if not _is_number(k)]
    rows = []
    for k, r in ordered:
        s, t = r["summary"], r["summary"]["test"]
        sw_on, dur_on = s["lambda_switch"] > 0, s["lambda_duration"] > 0
        weights = "yes" if str(s.get("switch_weight", "balanced")) != "none" else "no"
        log = r["train_log"]
        tail = log.tail(max(len(log) // 10, 1)) if log is not None else None
        fmt = lambda v, on=True: f"{v:.3f}" if on else "n/a"  # noqa: E731
        rows.append([k, f"{s['lambda_switch']:g}", f"{s['lambda_duration']:g}", weights,
                     fmt(t["mean_F1"], sw_on), fmt(t["sigma_F1"], sw_on), fmt(t["mean_F1_at_0.5"], sw_on),
                     fmt(t["mean_dur_acc"], dur_on), fmt(t["mean_F1_macro_by_type"], sw_on),
                     fmt(tail["loss_switch"].mean(), sw_on) if tail is not None else "",
                     fmt(tail["loss_duration"].mean(), dur_on) if tail is not None else ""])
    out_md.write_text("# Loss weighting (Model A, 2,000 training conversations per pair, test split)\n\n"
                      + markdown_table(header, rows) + "\n", encoding="utf-8")


def parse_named(items: list[str]) -> dict[str, dict]:
    out = {}
    for item in items or []:
        name, _, path = item.partition("=")
        out[name] = load_run(Path(path))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help='"display name=run_dir" ...')
    ap.add_argument("--sweep", nargs="*", default=[],
                    help='"lambda_duration value=run_dir" ... plus named extra runs, e.g. "no class weights=runs/sweep_lam1_noweight"')
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="baseline_f1.csv of Update 1 ('' to omit)")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(args.baseline) if args.baseline and Path(args.baseline).exists() else None

    runs = parse_named(args.runs)
    for name, run in runs.items():
        if run["train_log"] is not None:
            plot_convergence(name, run, out / f"convergence_{slug(name)}.png")
    if len(runs) > 1:
        plot_convergence_compare(runs, out / "convergence_compare.png")
    if runs:
        plot_f1_by_pair(runs, baseline, out / "f1_by_pair.png", out / "f1_by_pair.md")
        comparison_table(runs, baseline, out / "comparison.md")
    sweep = parse_named(args.sweep)
    if sweep:
        plot_sweep(sweep, out / "lambda_sweep.png", out / "lambda_sweep.md")
    print(f"wrote figures and tables to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
