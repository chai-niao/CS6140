"""Kaggle kernel (script) that trains and evaluates the Update 2 models on the free T4 x2 GPUs.

Generated from kaggle/run_kaggle_template.py by kaggle/stage.py; RUNS below is filled from a plan file.
Inputs (private Kaggle datasets):
    /kaggle/input/switchlingua-processed     <pair>__{train,dev,test}.jsonl written by update1/split.py
    /kaggle/input/cs6140-update2-code        code.zip with update1/ and update2/ (or the unzipped folders)
Outputs: /kaggle/working/runs/<run name>/...  (downloaded with `kaggle kernels output`)
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

RUNS = __RUNS__
KEEP_CHECKPOINTS = __KEEP_CHECKPOINTS__
STREAMING_CHECK = __STREAMING_CHECK__

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CODE_DIR = WORK / "code"
RUNS_DIR = WORK / "runs"


def find_input(slug: str, marker: str) -> Path:
    """Locate a mounted dataset. Kaggle has used both /kaggle/input/<slug> and nested
    layouts such as /kaggle/input/datasets/<owner>/<slug>, so search for a marker file."""
    direct = INPUT / slug
    if direct.exists() and any(direct.rglob(marker)):
        return next(direct.rglob(marker)).parent
    for hit in sorted(INPUT.rglob(marker)):
        if slug in hit.as_posix():
            return hit.parent
    raise FileNotFoundError(f"dataset {slug} with {marker} not found under {INPUT}")


DATA_DIR = find_input("switchlingua-processed", "*__train.jsonl")


def sh(cmd: list[str], **kw) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, **kw)


def install_requirements() -> None:
    # update1/lid.py (imported through dataset.py) needs the taggers' libraries
    sh([sys.executable, "-m", "pip", "install", "-q", "wordfreq", "jieba", "pycantonese", "fugashi", "unidic-lite", "sentencepiece", "protobuf"])
    import transformers  # noqa: WPS433

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (4, 40):
        sh([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.40"])


def unpack_code() -> None:
    if CODE_DIR.exists():
        shutil.rmtree(CODE_DIR)
    CODE_DIR.mkdir(parents=True)
    zips = [z for z in INPUT.rglob("code.zip") if "cs6140-update2-code" in z.as_posix()]
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(CODE_DIR)
    else:  # Kaggle unpacked the archive already
        src = find_input("cs6140-update2-code", "train.py").parent
        for d in ("update1", "update2"):
            shutil.copytree(src / d, CODE_DIR / d)
    print("code:", sorted(p.name for p in (CODE_DIR / "update2").iterdir()), flush=True)


def main() -> None:
    t0 = time.time()
    print("GPU:", subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    install_requirements()
    unpack_code()
    print("data dir:", DATA_DIR, "files:", len(list(DATA_DIR.glob("*.jsonl"))), flush=True)
    RUNS_DIR.mkdir(exist_ok=True)
    summaries = []
    for run in RUNS:
        name = run["name"]
        out_dir = RUNS_DIR / name
        cmd = [sys.executable, "train.py", "--data_dir", str(DATA_DIR), "--out_dir", str(out_dir), "--run_name", name] + run["args"]
        print(f"\n===== run {name}  ({(time.time() - t0) / 60:.0f} min elapsed) =====", flush=True)
        code = sh(cmd, cwd=CODE_DIR / "update2")
        if code != 0:
            print(f"run {name} failed with exit code {code}", flush=True)
            continue
        if STREAMING_CHECK and (out_dir / "model.pt").exists():
            sh([sys.executable, "evaluate.py", "--run_dir", str(out_dir), "--split", "test", "--max_convs", "20",
                "--streaming_check", str(STREAMING_CHECK)], cwd=CODE_DIR / "update2")
        if not KEEP_CHECKPOINTS and (out_dir / "model.pt").exists():
            (out_dir / "model.pt").unlink()
        summaries.append(json.loads((out_dir / "summary.json").read_text()))
    print("\n===== all runs =====")
    for s in summaries:
        t = s["test"]
        print(f"{s['run_name']:28s} steps {s['steps']:>6}  {s['train_minutes']:>6.1f} min  mean F1 {t['mean_F1']:.4f}  "
              f"sigma {t['sigma_F1']:.4f}  dur acc {t['mean_dur_acc']:.4f}")
    (RUNS_DIR / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    shutil.rmtree(CODE_DIR, ignore_errors=True)  # keep only runs/ in the kernel output
    print(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
