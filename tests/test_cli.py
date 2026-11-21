import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from jena_image_render import cli, env


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cwd = os.getcwd()
        os.chdir(self.root)
        os.environ["JENA_COMFY_URL"] = "http://127.0.0.1:9"  # nothing listens here
        (self.root / "models/checkpoints").mkdir(parents=True)
        (self.root / "models/checkpoints/ill.safetensors").write_bytes(b"x")
        (self.root / "job.json").write_text(json.dumps(
            {"version": 1, "defaults": {"model": "ill", "seeds": [1, 2]}, "jobs": [{"name": "a", "prompt": "x"}]}
        ))

    def tearDown(self) -> None:
        os.environ.pop("JENA_COMFY_URL", None)
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertLogs("jena_image_render", "INFO") as logs:
            cli.logging.getLogger("jena_image_render").info("start")
            code = cli.main(list(argv))
        return code, out.getvalue(), "\n".join(logs.output)

    def test_dry_run_reports_plan(self) -> None:
        code, out, _ = self.run_cli("job.json", "--runpod", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("2 image(s) to render", out)
        self.assertIn("upload", out)
        self.assertIn("job/ill/a_2.png", out)

    def test_dry_run_fails_on_a_local_file_that_vanished(self) -> None:
        real = cli.assets.plan

        def vanished(files, target, storage=False):
            Path("models/checkpoints/ill.safetensors").unlink()
            return real(files, target, storage)

        cli.assets.plan = vanished
        try:
            code, out, _ = self.run_cli("job.json", "--runpod", "--dry-run")
        finally:
            cli.assets.plan = real
        self.assertEqual(code, 1)
        self.assertIn("missing local files", out)

    def test_local_is_the_default_target(self) -> None:
        code, out, logs = self.run_cli("job.json", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("files (local ComfyUI):", out)
        self.assertIn("ComfyUI is not answering on http://127.0.0.1:9", logs)
        self.assertNotIn("WARNING", logs)  # an offline check is not a problem

    def test_default_target_reaches_for_the_local_comfyui(self) -> None:
        code, _, logs = self.run_cli("job.json")
        self.assertEqual(code, 1)
        self.assertIn("ComfyUI is not answering", logs)
        self.assertIn("--runpod", logs)

    def test_config_errors_exit_1(self) -> None:
        Path("job.json").write_text(json.dumps({"version": 1, "model": "nope", "prompt": "x"}))
        code, _, logs = self.run_cli("job.json", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("model 'nope' is not a known alias", logs)
        self.assertIn("models/checkpoints/", logs)
        self.assertIn("models/diffusion_models/<engine>/", logs)
        self.assertIn('"model": "hf://', logs)
        self.assertNotIn("models.json", logs)

    def write_job(self, file: str, **extra) -> None:
        Path(file).write_text(json.dumps({"version": 1, "model": "ill", "prompt": "x", **extra}))

    def test_target_precedence(self) -> None:
        self.write_job("job.json", target="local")
        code, out, _ = self.run_cli("job.json", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("files (local ComfyUI):", out)
        code, out, _ = self.run_cli("job.json", "--runpod", "--dry-run")  # command line wins
        self.assertIn("files (RunPod):", out)
        self.assertIn("upload ", out)
        self.write_job("job.json")
        os.environ["JENA_TARGET"] = "local"
        try:
            code, out, _ = self.run_cli("job.json", "--dry-run")
            self.assertIn("files (local ComfyUI):", out)
            self.write_job("job.json", target="runpod")  # the job file beats .env
            code, out, _ = self.run_cli("job.json", "--dry-run")
            self.assertIn("files (RunPod):", out)
            os.environ["JENA_TARGET"] = "cloud"
            self.write_job("job.json")
            code, _, logs = self.run_cli("job.json", "--dry-run")
            self.assertEqual(code, 1)
            self.assertIn("JENA_TARGET must be", logs)
        finally:
            os.environ.pop("JENA_TARGET", None)

    def test_conflicting_targets_need_the_command_line(self) -> None:
        self.write_job("a.json", target="local", name="a")
        self.write_job("b.json", target="runpod", name="b")
        code, _, logs = self.run_cli("a.json", "b.json", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("different targets", logs)
        code, out, _ = self.run_cli("a.json", "b.json", "--local", "--dry-run")
        self.assertEqual(code, 0)

    def test_bad_target_value(self) -> None:
        self.write_job("job.json", target="cloud")
        code, _, logs = self.run_cli("job.json", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn('target must be "local" or "runpod"', logs)

    def test_storage_changes_the_runpod_plan(self) -> None:
        os.environ["JENA_STORAGE"] = "hf://someone/models"
        try:
            code, out, _ = self.run_cli("job.json", "--runpod", "--dry-run")
        finally:
            os.environ.pop("JENA_STORAGE", None)
        self.assertEqual(code, 0)
        self.assertIn("storage ", out)

    def test_builtin_model_needs_nothing(self) -> None:
        self.write_job("job.json", model="zimage")
        code, out, _ = self.run_cli("job.json", "--runpod", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("fetch   https://huggingface.co/Comfy-Org/Z-Image-Turbo/resolve/", out)

    def test_folder_checkpoint_is_used_by_name(self) -> None:
        Path("models/checkpoints/my-anime.safetensors").write_bytes(b"w")
        self.write_job("job.json", model="my-anime")
        code, out, _ = self.run_cli("job.json", "--runpod", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("upload  models/checkpoints/my-anime.safetensors → models/checkpoints/my-anime.safetensors", out)

    def test_push_needs_storage_and_pull_needs_known_names(self) -> None:
        code, _, logs = self.run_cli("push", "job.json")
        self.assertEqual(code, 1)
        self.assertIn("JENA_STORAGE", logs)
        self.write_job("job.json", model="zimage")
        code, _, logs = self.run_cli("push", "job.json")
        self.assertEqual(code, 0)
        self.assertIn("Nothing to push", logs)
        code, _, logs = self.run_cli("pull", "nope")
        self.assertEqual(code, 1)
        self.assertIn("not a model or LoRA alias", logs)

    def test_addresses_in_the_job(self) -> None:
        self.write_job(
            "job.json",
            model="hf://someone/their-model/sub/their-ckpt.safetensors",
            loras=[{"name": "https://cdn.example.com/x.safetensors", "scale": 0.5}],
        )
        code, out, _ = self.run_cli("job.json", "--runpod", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("job/their-ckpt/job1_0.png", out)
        self.assertIn(
            "fetch   https://huggingface.co/someone/their-model/resolve/main/sub/"
            "their-ckpt.safetensors → models/checkpoints/_url/", out
        )
        self.assertIn("https://cdn.example.com/x.safetensors → models/loras/_url/", out)

    def test_models_option_is_gone(self) -> None:
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli.main(["job.json", "--models", "models.json", "--dry-run"])

    def test_engines_listing(self) -> None:
        code, out, _ = self.run_cli("engines")
        self.assertEqual(code, 0)
        self.assertIn("illustrious", out)
        self.assertIn("2:3=832x1248", out)


class DotenvTests(unittest.TestCase):
    def test_reads_values_without_overriding_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text('# c\nJENA_T1="quoted"\nexport JENA_T2=plain\nJENA_T3=fromfile\n')
            os.environ["JENA_T3"] = "real"
            try:
                env.load_dotenv(path)
                self.assertEqual(os.environ["JENA_T1"], "quoted")
                self.assertEqual(os.environ["JENA_T2"], "plain")
                self.assertEqual(os.environ["JENA_T3"], "real")
            finally:
                for key in ("JENA_T1", "JENA_T2", "JENA_T3"):
                    os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
