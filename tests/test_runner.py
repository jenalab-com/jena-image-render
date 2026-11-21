import json
import tempfile
import unittest
from pathlib import Path

from helpers import registry

from jena_image_render import runner
from jena_image_render.comfy import ComfyError
from jena_image_render.config import expand
from jena_image_render.graph import Names


class FakeClient:
    def __init__(self, fail_on=()) -> None:
        self.fail_on = set(fail_on)
        self.graphs = []

    def run(self, graph, output_node, on_step=None):
        self.graphs.append(graph)
        if len(self.graphs) in self.fail_on:
            raise ComfyError("boom")
        if on_step:
            on_step(1, 1)
        return b"PNG" + str(len(self.graphs)).encode()


def batch():
    return expand(
        {"version": 1, "output": "o", "defaults": {"model": "zimage", "seeds": [1, 2]},
         "jobs": [{"name": "a", "prompt": "x"}, {"name": "b", "prompt": "y"}]},
        registry(), "t",
    )


NAMES = lambda r: Names("b", "c", "v")  # noqa: E731


class RunnerTests(unittest.TestCase):
    def test_select_skips_saved_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "o/zimage").mkdir(parents=True)
            (out / "o/zimage/a_1.png").write_bytes(b"x")
            todo, skipped = runner.select(batch(), out)
            self.assertEqual((len(todo), skipped), (3, 1))
            todo, _ = runner.select(batch(), out, overwrite=True)
            self.assertEqual(len(todo), 4)
            todo, _ = runner.select(batch(), out, only="b", limit=1)
            self.assertEqual([r.output for r in todo], ["o/zimage/b_1.png"])

    def test_render_all_saves_image_and_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = runner.render_all(FakeClient(), batch(), NAMES, out, where="local")
            self.assertEqual(summary.done, 4)
            self.assertEqual((out / "o/zimage/b_2.png").read_bytes(), b"PNG4")
            meta = json.loads((out / "o/zimage/b_2.json").read_text())
            self.assertEqual((meta["seed"], meta["prompt"], meta["server"]), (2, "y", "local"))
            self.assertFalse(list(out.rglob("*.part")))

    def test_stops_on_error_unless_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("jena_image_render.runner", "ERROR"):
                summary = runner.render_all(FakeClient({2}), batch(), NAMES, Path(tmp))
            self.assertEqual((summary.done, summary.failed), (1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("jena_image_render.runner", "ERROR"):
                summary = runner.render_all(FakeClient({2}), batch(), NAMES, Path(tmp), keep_going=True)
            self.assertEqual((summary.done, summary.failed), (3, 1))

    def test_describe_lists_files_and_outputs(self) -> None:
        todo = batch()
        from jena_image_render import assets

        plan = assets.plan(assets.needed(todo, registry()), "runpod")
        text = runner.describe(todo, 1, plan, "runpod")
        self.assertIn("4 image(s) to render, 1 already saved", text)
        self.assertIn("files (RunPod):", text)
        self.assertIn("fetch   https://huggingface.co/org/zimage/resolve/rev1/split/z.safetensors", text)
        self.assertIn("o/zimage/a_1.png", text)

    def test_describe_local_plan(self) -> None:
        from jena_image_render import assets

        todo = batch()
        plan = assets.plan(assets.needed(todo, registry()), "local")
        text = runner.describe(todo, 0, plan, "local")
        self.assertIn("files (local ComfyUI):", text)
        self.assertIn("local   models/diffusion_models/z.safetensors (not in your models folder: downloaded if missing)", text)
        self.assertIn("ComfyUI did not answer", text)
        auto = runner.describe(todo, 0, [("have", plan[0][1])], "local", {})
        self.assertNotIn("did not answer", auto)
        self.assertIn("have     models/", auto)


if __name__ == "__main__":
    unittest.main()
