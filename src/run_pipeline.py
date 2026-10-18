
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Data is already downloaded, so this pipeline starts with cleaning.
PIPELINE_STAGES = [
    ("Data cleaning", "data_cleaning.py"),
    ("Pair selection", "pair_selection.py"),
    ("Trading signals", "signals.py"),
    ("Backtesting", "backtest.py"),
    ("Interactive visualizations", "visualizations.py"),
]


def run_stage(name, script):
    script_path = SRC / script

    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    print(f"\n{'=' * 65}", flush=True)
    print(f"STARTING: {name}", flush=True)
    print(f"Script: {script}", flush=True)
    print(f"{'=' * 65}", flush=True)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit code {result.returncode}. "
            "Fix this stage before continuing."
        )

    print(f"COMPLETED: {name}", flush=True)


def main():
    start_time = datetime.now()

    print("MEAN-REVERSION PAIRS TRADING PIPELINE", flush=True)
    print(f"Project directory: {ROOT}", flush=True)
    print(f"Python executable: {sys.executable}", flush=True)

    try:
        for name, script in PIPELINE_STAGES:
            run_stage(name, script)

    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nPIPELINE STOPPED: {exc}", flush=True)
        sys.exit(1)

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 65, flush=True)
    print("PIPELINE COMPLETED SUCCESSFULLY", flush=True)
    print(f"Elapsed time: {elapsed:.1f} seconds", flush=True)
    print(f"Charts directory: {ROOT / 'reports' / 'charts'}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
