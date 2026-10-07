"""Run supported methods sequentially in isolated processes, then compare results."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __name__ == "__main__": print("Loading baseline dependencies...", flush=True)

import numpy as np

from piu_unlearning.config import RunConfig, parse_config
from piu_unlearning.data import create_evaluation_conditions, create_experiment_split, load_prepared_data, save_evaluation_conditions, select_anchor_embedding, write_split_manifest
from piu_unlearning.dataset import prepare_for_demo
from piu_unlearning.models.arc2face import GeneratedSamples
from piu_unlearning.methods.siss import prepare_siss_inputs
from piu_unlearning.methods.wid import check_wid_inputs
from piu_unlearning.visualization import write_method_comparison


METHODS = ("piu", "siss", "uce", "wid")
METRICS = ("forget_ism", "retain_ism", "AccU", "AccR", "SRK")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-id", type=int, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--data-dir", type=Path, default=RunConfig.data_dir)
    for name in ("device", "evaluation_device"):
        parser.add_argument(f"--{name.replace('_', '-')}", default=getattr(RunConfig, name))
    for name in ("seed", "num_samples", "num_inference_steps"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(RunConfig, name))
    for name in ("guidance_scale", "forget_validation_ratio", "retain_validation_ratio", "proximity_threshold", "anchor_tolerance"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(RunConfig, name))
    parser.add_argument("--use-anchor-overrides", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without loading models, data, or writing outputs.")
    for method in METHODS: parser.add_argument(f"--{method}-args", default="", help=f"Quoted method-specific piu-demo options for {method.upper()}.")
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods): parser.error("Each method must appear only once")
    args.output_dir, args.data_dir = args.output_dir.resolve(), args.data_dir.resolve()
    return args


def build_jobs(args):
    common = []
    for name in ("identity_id", "data_dir", "device", "evaluation_device", "seed", "num_samples", "num_inference_steps", "guidance_scale", "forget_validation_ratio", "retain_validation_ratio"):
        common.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    anchor = ["--proximity-threshold", str(args.proximity_threshold), "--anchor-tolerance", str(args.anchor_tolerance)]
    if args.use_anchor_overrides: anchor.append("--use-anchor-overrides")
    split_args = [*common, *anchor, "--output-dir", str(args.output_dir)]
    parse_config(split_args)  # SISS-only runs still apply anchor overrides to the shared split.
    jobs = []
    for index, method in enumerate(args.methods):
        command = ["--method", method, *common, "--output-dir", str(args.output_dir / method)]
        if method != "siss": command.extend(anchor)
        if index: command.append("--reuse-baseline")
        expected = parse_config(command)
        command.extend(shlex.split(getattr(args, f"{method}_args")))
        config = parse_config(command)
        # Per-method overrides must not change what is being compared or where it is saved.
        for name in RunConfig.__dataclass_fields__:
            if name != "surgical_layers" and getattr(config, name) != getattr(expected, name):
                raise ValueError(f"{method}: --{name.replace('_', '-')} is shared; set it on the launcher instead")
        if config.method != method: raise ValueError("Method overrides cannot change --method")
        jobs.append({"method": method, "args": command, "split_args": split_args})
    return jobs


def prepare_split(split_args):
    config = parse_config(split_args)
    print(f"Loading prepared data from {config.data_dir}...", flush=True)
    data = load_prepared_data(config)
    print("Preparing training splits and evaluation conditions...", flush=True)
    split = create_experiment_split(*data, config)
    return split, create_evaluation_conditions(split, config)


def worker(path):
    from piu_unlearning.main import run_demo

    job = json.loads(path.read_text())
    split, _ = prepare_split(job["split_args"])
    run_demo(parse_config(job["args"]), split=split)


def run_process(command, log_path):
    """Keep normal terminal progress and a complete combined stdout/stderr log."""
    with log_path.open("w", encoding="utf-8") as log, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if code: raise RuntimeError(f"Method exited with code {code}; see {log_path}")


def verify_run(output_dir, method):
    method_dir = output_dir / method
    if json.loads((method_dir / "split.json").read_text()) != json.loads((output_dir / "split.json").read_text()):
        raise ValueError(f"{method}: split differs from the shared comparison split")
    with np.load(output_dir / "evaluation_conditions.npz") as shared, np.load(method_dir / "evaluation_conditions.npz") as actual:
        if set(shared.files) != set(actual.files) or any(not np.array_equal(shared[key], actual[key]) for key in shared.files):
            raise ValueError(f"{method}: evaluation conditions differ from the shared comparison conditions")
    summary = json.loads((method_dir / "summary.json").read_text())
    if summary["method"] != method: raise ValueError(f"{method}: unexpected summary method")
    return summary


def metric_values(phase):
    return dict(zip(METRICS, (phase["forget"]["ism"], phase["retain"]["ism"], phase["srk"]["forget_accuracy"], phase["srk"]["retain_accuracy"], phase["srk"]["score"])))


def write_comparison(output_dir, summaries, timings, conditions):
    baseline = metric_values(summaries[0]["evaluation"]["before"])
    rows = [{"method": "original", **baseline, **{f"delta_{key}": 0.0 for key in METRICS}, "elapsed_seconds": ""}]
    samples = {"original": GeneratedSamples(**{key: [Path(path) for path in paths] for key, paths in summaries[0]["before_images"].items()})}
    for summary in summaries:
        method = summary["method"]
        before = metric_values(summary["evaluation"]["before"])
        if not np.allclose(list(before.values()), list(baseline.values()), rtol=1e-5, atol=1e-6):
            raise ValueError(f"{method}: original-model metrics disagree despite shared baseline images")
        metrics = metric_values(summary["evaluation"]["after"])
        rows.append({"method": method, **metrics, **{f"delta_{key}": metrics[key] - baseline[key] for key in METRICS}, "elapsed_seconds": timings[method]})
        samples[method] = GeneratedSamples(**{key: [Path(path) for path in paths] for key, paths in summary["after_images"].items()})
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    report = ["# Method comparison", "", "| Method | Forget ISM | Retain ISM | AccU | AccR | SRK |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows: report.append(f"| {row['method']} | " + " | ".join(f"{row[key]:.4f}" for key in METRICS) + " |")
    report.extend(["", "Lower forget ISM and AccU indicate stronger forgetting; higher retain ISM, AccR, and SRK indicate better results. Inspect both forgetting and preservation, not SRK alone.", "", "CSV/JSON include changes from the original model. Elapsed time is total subprocess wall time, including loading, generation, and evaluation; it is not an editing-speed benchmark. Methods keep their own learning defaults.", "", "![Forget comparison](comparison_forget.png)", "", "![Retain comparison](comparison_retain.png)"])
    (output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_method_comparison(samples, conditions, output_dir)
    print("\n".join(report[:4 + len(rows)]), flush=True)


def launch(args):
    jobs = build_jobs(args)
    if args.dry_run:
        for job in jobs:
            print(f"\n{job['method'].upper()}: {shlex.join(job['args'])}")
            print(json.dumps(asdict(parse_config(job["args"])), indent=2, default=str))
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise ValueError("Output directory is not empty; choose a new --output-dir to avoid mixing runs")
    for job in jobs: prepare_for_demo(parse_config(job["args"]))
    split, conditions = prepare_split(jobs[0]["split_args"])
    if any(job["method"] != "siss" for job in jobs):
        print("Checking the shared anchor...", flush=True)
        select_anchor_embedding(split, parse_config(jobs[0]["split_args"]))
    for job in jobs:
        print(f"Checking {job['method'].upper()} training inputs...", flush=True)
        if job["method"] == "siss": prepare_siss_inputs(split, parse_config(job["args"]))
        if job["method"] == "wid": check_wid_inputs(split, parse_config(job["args"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(split, args.output_dir / "split.json")
    save_evaluation_conditions(conditions, args.output_dir / "evaluation_conditions.npz")
    summaries, timings, status = [], {}, []
    for job in jobs:
        method_dir = args.output_dir / job["method"]
        method_dir.mkdir()
        if summaries: shutil.copytree(args.output_dir / jobs[0]["method"] / "before", method_dir / "before")
        job_path = method_dir / "job.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        entry = {"method": job["method"], "status": "running", "log": str(method_dir / "run.log")}
        status.append(entry)
        status_path = args.output_dir / "runs.json"
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(f"Starting {job['method'].upper()}; log: {method_dir / 'run.log'}", flush=True)
        start = time.monotonic()
        try:
            run_process([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(job_path)], method_dir / "run.log")
            summaries.append(verify_run(args.output_dir, job["method"]))
            entry["status"] = "completed"
        except (Exception, KeyboardInterrupt) as error:
            entry.update(status="failed", error=str(error) or "Interrupted")
            raise
        finally:
            timings[job["method"]] = entry["elapsed_seconds"] = time.monotonic() - start
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    write_comparison(args.output_dir, summaries, timings, conditions)
    print(f"Comparison written to {args.output_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker": worker(Path(sys.argv[2]))
    else: launch(parse_args())
