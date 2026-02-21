import subprocess
import sys
import os
import time
import argparse


def run_step(name, cmd):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")
    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {name} ({elapsed:.1f}s)")
    if result.returncode != 0:
        print(f"  Command failed with return code {result.returncode}")
        sys.exit(1)
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-phase", action="store_true")
    parser.add_argument("--skip-spectral", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    os.makedirs("results", exist_ok=True)

    total_start = time.time()
    timings = {}

    if args.quick:
        phase_args = "--n-base 30 --num-seeds 5 --num-gamma 6 --num-delta 4 --epochs 1000"
        spectral_args = "--n-base 20 --num-seeds 3 --num-delta 3 --epochs 1500"
    else:
        phase_args = "--n-base 50 --num-seeds 10 --num-gamma 8 --num-delta 5 --epochs 2000"
        spectral_args = "--n-base 30 --num-seeds 5 --num-delta 4 --epochs 3000"

    if not args.skip_phase:
        timings["phase_boundary"] = run_step(
            "Phase Boundary Experiment",
            f"{sys.executable} phase_boundary.py {phase_args}",
        )

    if not args.skip_spectral:
        timings["spectral_gap"] = run_step(
            "Spectral Gap Experiment",
            f"{sys.executable} spectral_gap.py {spectral_args}",
        )

    if not args.skip_plot:
        timings["plotting"] = run_step(
            "Generating Figures",
            f"{sys.executable} plot_results.py",
        )

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}")
    for step, t in timings.items():
        print(f"  {step:25s} {t:7.1f}s")
    print(f"  {'total':25s} {total_elapsed:7.1f}s")
    print(f"\nResults in: {os.path.join(script_dir, 'results')}")
    print(f"Figures in: {os.path.join(script_dir, '..', 'paper', 'figures')}")


if __name__ == "__main__":
    main()
