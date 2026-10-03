import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from piu_unlearning.config import parse_config
from piu_unlearning.data import create_evaluation_conditions, save_evaluation_conditions, write_split_manifest
from test_methods import make_split


spec = importlib.util.spec_from_file_location("baseline_launcher", Path(__file__).resolve().parents[1] / "baselines/run_all.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_defaults_keep_method_settings_and_shared_split(self):
        args = launcher.parse_args(["--identity-id", "512", "--use-anchor-overrides"])
        jobs = launcher.build_jobs(args)
        self.assertEqual([job["method"] for job in jobs], ["piu", "siss", "uce", "wid"])
        piu, siss, uce, wid = [parse_config(job["args"]) for job in jobs]
        self.assertEqual((wid.training_steps, wid.identity_loss_weight), (100, 0.1))
        self.assertEqual((piu.training_steps, siss.training_steps), (400, 60))
        self.assertEqual((piu.optimizer, siss.optimizer), ("adamw", "adamw"))
        self.assertFalse(hasattr(uce, "training_steps"))
        self.assertEqual((piu.reuse_baseline, siss.reuse_baseline, uce.reuse_baseline), (False, True, True))
        self.assertIsNone(siss.anchor_overrides)
        self.assertIsNotNone(parse_config(jobs[1]["split_args"]).anchor_overrides)
        self.assertTrue(all(job["split_args"] == jobs[0]["split_args"] for job in jobs))

    def test_method_overrides_and_comparison_guards(self):
        args = launcher.parse_args(["--identity-id", "512", "--piu-args=--training-steps 2 --evaluation-every 0", "--siss-args=--training-steps 3", "--uce-args=--erase-scale 10"])
        configs = [parse_config(job["args"]) for job in launcher.build_jobs(args)]
        self.assertEqual((configs[0].training_steps, configs[1].training_steps, configs[2].erase_scale), (2, 3, 10))
        for option in ("--seed 1", "--data-dir elsewhere", "--num-samples 1", "--output-dir somewhere", "--base-model other"):
            args.piu_args = option
            with self.subTest(option=option), self.assertRaises(ValueError): launcher.build_jobs(args)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(["--identity-id", "512", "--methods", "piu", "piu"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            launcher.parse_args(["--identity-id", "512", "--methods", "siss", "--use-anchor-overrides"])

    def test_dry_run_does_not_load_data_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            args = launcher.parse_args(["--identity-id", "512", "--output-dir", str(Path(directory) / "run"), "--dry-run"])
            with patch.object(launcher, "prepare_split", side_effect=AssertionError("No data needed")), contextlib.redirect_stdout(io.StringIO()): launcher.launch(args)
            self.assertFalse(args.output_dir.exists())

    def test_process_logging_and_failure(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            log = Path(directory) / "run.log"
            launcher.run_process([sys.executable, "-u", "-c", "import sys; print('out'); print('err', file=sys.stderr)"], log)
            self.assertEqual(log.read_text().splitlines(), ["out", "err"])
            with self.assertRaisesRegex(RuntimeError, "code 3"):
                launcher.run_process([sys.executable, "-c", "raise SystemExit(3)"], log)

    def test_worker_passes_shared_split_to_existing_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            job = launcher.build_jobs(launcher.parse_args(["--identity-id", "512", "--methods", "siss"]))[0]
            path = Path(directory) / "job.json"
            path.write_text(json.dumps(job))
            split = object()
            with patch.object(launcher, "prepare_split", return_value=(split, None)), patch("piu_unlearning.main.run_demo") as run_demo:
                launcher.worker(path)
            self.assertEqual(run_demo.call_args.args[0].method, "siss")
            self.assertIs(run_demo.call_args.kwargs["split"], split)

    def test_sequential_runs_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = launcher.parse_args(["--identity-id", "0", "--device", "cpu", "--num-samples", "1", "--output-dir", str(Path(directory) / "comparison")])
            split = make_split(parse_config(["--identity-id", "0"]))
            conditions = create_evaluation_conditions(split, parse_config(["--identity-id", "0", "--num-samples", "1"]))
            calls = []

            def fake_process(command, log_path):
                job = json.loads(Path(command[-1]).read_text())
                config = parse_config(job["args"])
                calls.append(config.method)
                log_path.write_text("mock run\n")
                write_split_manifest(split, config.output_dir / "split.json")
                save_evaluation_conditions(conditions, config.output_dir / "evaluation_conditions.npz")
                images = {}
                for phase in ("before", "after"):
                    images[f"{phase}_images"] = {}
                    for partition in ("forget", "retain"):
                        path = config.output_dir / phase / partition / "0000.png"
                        if phase == "before" and config.reuse_baseline:
                            self.assertTrue(path.is_file())
                            self.assertEqual(path.read_bytes(), (args.output_dir / "piu" / phase / partition / "0000.png").read_bytes())
                        else:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            Image.new("RGB", (16, 16), "white" if phase == "before" else "blue").save(path)
                        images[f"{phase}_images"][partition] = [str(path)]
                before = {"forget": {"ism": 0.8}, "retain": {"ism": 0.7}, "srk": {"forget_accuracy": 1.0, "retain_accuracy": 1.0, "score": 0.99}}
                after = {"forget": {"ism": 0.3}, "retain": {"ism": 0.6}, "srk": {"forget_accuracy": 0.1, "retain_accuracy": 0.9, "score": 8.18}}
                (config.output_dir / "summary.json").write_text(json.dumps({"method": config.method, "evaluation": {"before": before, "after": after}, **images}))

            with patch.object(launcher, "prepare_for_demo"), patch.object(launcher, "prepare_split", return_value=(split, conditions)), patch.object(launcher, "select_anchor_embedding"), patch.object(launcher, "prepare_siss_inputs"), patch.object(launcher, "check_wid_inputs"), patch.object(launcher, "run_process", side_effect=fake_process), contextlib.redirect_stdout(io.StringIO()): launcher.launch(args)
            self.assertEqual(calls, ["piu", "siss", "uce", "wid"])
            for name in ("comparison.csv", "comparison.json", "comparison.md", "comparison_forget.png", "comparison_retain.png"):
                self.assertTrue((args.output_dir / name).is_file())
            rows = json.loads((args.output_dir / "comparison.json").read_text())
            self.assertEqual([row["method"] for row in rows], ["original", "piu", "siss", "uce", "wid"])
            self.assertAlmostEqual(rows[1]["delta_forget_ism"], -0.5)
            self.assertEqual([run["status"] for run in json.loads((args.output_dir / "runs.json").read_text())], ["completed"] * 4)
            with self.assertRaisesRegex(ValueError, "not empty"): launcher.launch(args)
            (args.output_dir / "uce/split.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "split differs"): launcher.verify_run(args.output_dir, "uce")
            write_split_manifest(split, args.output_dir / "uce/split.json")
            np = launcher.np
            np.savez(args.output_dir / "uce/evaluation_conditions.npz", changed=np.array([1]))
            with self.assertRaisesRegex(ValueError, "conditions differ"): launcher.verify_run(args.output_dir, "uce")

    def test_failed_run_stops_remaining_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            args = launcher.parse_args(["--identity-id", "0", "--num-samples", "1", "--output-dir", str(Path(directory) / "run")])
            config = parse_config(["--identity-id", "0", "--num-samples", "1"])
            split = make_split(config)
            with patch.object(launcher, "prepare_for_demo"), patch.object(launcher, "prepare_split", return_value=(split, create_evaluation_conditions(split, config))), patch.object(launcher, "select_anchor_embedding"), patch.object(launcher, "prepare_siss_inputs"), patch.object(launcher, "check_wid_inputs"), patch.object(launcher, "run_process", side_effect=RuntimeError("failed")) as process:
                with self.assertRaisesRegex(RuntimeError, "failed"): launcher.launch(args)
            self.assertEqual(process.call_count, 1)
            self.assertEqual(json.loads((args.output_dir / "runs.json").read_text())[0]["status"], "failed")
            self.assertFalse((args.output_dir / "comparison.csv").exists())

    def test_siss_preflight_runs_without_anchor_and_before_any_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            args = launcher.parse_args(["--identity-id", "0", "--methods", "siss", "--output-dir", str(Path(directory) / "run")])
            with patch.object(launcher, "prepare_for_demo"), patch.object(launcher, "prepare_split", return_value=(object(), object())), patch.object(launcher, "select_anchor_embedding", side_effect=AssertionError("No anchor needed")), patch.object(launcher, "prepare_siss_inputs", side_effect=FileNotFoundError("missing SISS images")), patch.object(launcher, "run_process") as process:
                with self.assertRaisesRegex(FileNotFoundError, "missing SISS images"): launcher.launch(args)
                process.assert_not_called()
            self.assertFalse(args.output_dir.exists())

    def test_wid_dependencies_checked_before_any_method_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = launcher.parse_args(["--identity-id", "0", "--output-dir", str(Path(directory) / "run"), "--wid-args=--identity-checkpoint weights.pt"])
            self.assertEqual(parse_config(launcher.build_jobs(args)[-1]["args"]).identity_checkpoint, Path("weights.pt"))
            with patch.object(launcher, "prepare_for_demo"), patch.object(launcher, "prepare_split", return_value=(object(), object())), patch.object(launcher, "select_anchor_embedding"), patch.object(launcher, "prepare_siss_inputs"), patch.object(launcher, "check_wid_inputs", side_effect=ValueError("missing WID images")), patch.object(launcher, "run_process") as process:
                with self.assertRaisesRegex(ValueError, "missing WID images"): launcher.launch(args)
                process.assert_not_called()
            self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
