"""Prepare the Kaggle uploads: two private datasets and one kernel per plan.

    python kaggle/stage.py --user <kaggle user> --stage_dir /tmp/kaggle_stage [--plan smoke]

Creates
    <stage_dir>/data/      dataset-metadata.json + <pair>__<split>.jsonl  (flat copy of update1/data_processed)
    <stage_dir>/code/      dataset-metadata.json + code.zip               (update1/*.py, update2/*.py, tests, baseline csv)
    <stage_dir>/kernel_<plan>/   kernel-metadata.json + run_kaggle.py     (RUNS filled from kaggle/plans/<plan>.json)

Then, with the Kaggle CLI configured (~/.kaggle/kaggle.json):
    kaggle datasets create -p <stage_dir>/data        (first time)   /  kaggle datasets version -p <stage_dir>/data -m "update"
    kaggle datasets create -p <stage_dir>/code                        /  kaggle datasets version -p <stage_dir>/code -m "update"
    kaggle kernels push -p <stage_dir>/kernel_<plan>
    kaggle kernels status <user>/cs6140-update2-<plan>
    kaggle kernels output <user>/cs6140-update2-<plan> -p runs_from_kaggle/<plan>
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPDATE2 = HERE.parent
UPDATE1 = UPDATE2.parent / "update1"
DATA_SLUG = "switchlingua-processed"
CODE_SLUG = "cs6140-update2-code"


def stage_data(user: str, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for pair_dir in sorted((UPDATE1 / "data_processed").iterdir()):
        if not (pair_dir / "train.jsonl").exists():
            continue
        for split in ("train", "dev", "test"):
            shutil.copy2(pair_dir / f"{split}.jsonl", out / f"{pair_dir.name}__{split}.jsonl")
            n += 1
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "SwitchLingua processed (CS6140 Update 2, private)",
        "id": f"{user}/{DATA_SLUG}",
        "licenses": [{"name": "other"}],
    }, indent=2))
    return n


def stage_code(user: str, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for d in (UPDATE1, UPDATE2):
        files += sorted(d.glob("*.py")) + sorted((d / "tests").glob("*.py")) + [d / "pytest.ini"]
    files += [UPDATE1 / "results" / "baseline_f1.csv", UPDATE2 / "requirements.txt", UPDATE1 / "requirements.txt",
              UPDATE2 / "duplicates.json"]
    with zipfile.ZipFile(out / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.exists():
                z.write(f, f.relative_to(UPDATE2.parent))
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "CS6140 Update 2 code (private)",
        "id": f"{user}/{CODE_SLUG}",
        "licenses": [{"name": "other"}],
    }, indent=2))
    return [str(f.relative_to(UPDATE2.parent)) for f in files if f.exists()]


def stage_kernel(user: str, plan_name: str, out: Path) -> None:
    plan = json.loads((HERE / "plans" / f"{plan_name}.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    template = (HERE / "run_kaggle_template.py").read_text()
    script = (template.replace("__RUNS__", json.dumps(plan["runs"], indent=4))
              .replace("__KEEP_CHECKPOINTS__", str(bool(plan.get("keep_checkpoints", False))))
              .replace("__STREAMING_CHECK__", str(int(plan.get("streaming_check", 0)))))
    (out / "run_kaggle.py").write_text(script)
    # Kaggle derives the kernel slug from the title and only allows dashes.
    slug = f"cs6140-update2-{plan_name.replace('_', '-')}"
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "run_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": plan.get("machine_shape", "NvidiaTeslaT4"),
        "dataset_sources": [f"{user}/{DATA_SLUG}", f"{user}/{CODE_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="Kaggle user name")
    ap.add_argument("--stage_dir", required=True)
    ap.add_argument("--plan", nargs="*", default=[], help="plan names from kaggle/plans/ (kernel folders to build)")
    ap.add_argument("--skip_data", action="store_true")
    args = ap.parse_args()
    stage = Path(args.stage_dir)
    if not args.skip_data:
        print(f"data:   {stage_data(args.user, stage / 'data')} files -> {stage / 'data'}")
    files = stage_code(args.user, stage / "code")
    print(f"code:   {len(files)} files in code.zip -> {stage / 'code'}")
    for plan in args.plan:
        stage_kernel(args.user, plan, stage / f"kernel_{plan}")
        print(f"kernel: {stage / f'kernel_{plan}'}")


if __name__ == "__main__":
    main()
