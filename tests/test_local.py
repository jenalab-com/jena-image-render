"""A ComfyUI on this computer: its folders, names found there, and filling them."""

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_pull import DATA, FakeOpener

from jena_image_render import assets, cli, comfy, local, models
from jena_image_render.folder import ComfyFolders, ModelsFolder
from jena_image_render.models import FileSpec

ADDRESS = "hf://someone/their-model/sub/their-ckpt.safetensors"
SHORT = hashlib.sha1(ADDRESS.encode("utf-8")).hexdigest()[:8]


def touch(path: Path, data: bytes = b"w") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def answer(c: Path, o: Path) -> dict[str, list[str]]:
    """Shaped like a real /internal/folder_paths answer: legacy aliases, folders
    from extra_model_paths.yaml and output folders, not all of them on disk."""
    return {
        "checkpoints": [f"{c}/models/checkpoints", f"{o}/models/illustrious",
                        f"{c}/output/checkpoints"],
        "diffusion_models": [f"{c}/models/unet", f"{c}/models/diffusion_models",
                             f"{o}/models/krea2", f"{c}/output/diffusion_models"],
        "text_encoders": [f"{c}/models/text_encoders", f"{c}/models/clip",
                          f"{c}/output/clip"],
        "vae": [f"{c}/models/vae", f"{c}/output/vae"],
        "loras": [f"{c}/models/loras", f"{o}/loras", f"{c}/output/loras"],
        "custom_nodes": [f"{c}/custom_nodes"],
    }


class FakeComfy:
    """Answers like a ComfyUI: lists files under every existing root of a role."""

    def __init__(self, paths=None, up=True) -> None:
        self.paths = paths
        self.up = up
        self.asked_paths = 0
        self.graphs: list[dict] = []

    def ready(self, timeout=15):
        return self.up

    def probe(self, timeout=15):
        return "200" if self.up else "ConnectionError"

    def folder_paths(self, timeout=15):
        self.asked_paths += 1
        return self.paths

    def options(self, folder):
        found = []
        for root in (self.paths or {}).get(folder, []):
            for top, _, names in os.walk(root, followlinks=True):
                for name in names:
                    rel = (Path(top) / name).relative_to(root).as_posix()
                    if rel not in found:
                        found.append(rel)
        return sorted(found)

    def run(self, graph, output_node, on_step=None):
        self.graphs.append(graph)
        return b"png"


class FakeSession:
    def __init__(self, status=200, body=None, error=None) -> None:
        self.status, self.body, self.error = status, body, error

    def get(self, url, timeout):
        if self.error:
            raise self.error
        body, status = self.body, self.status

        class Response:
            status_code = status

            def json(self):
                if isinstance(body, Exception):
                    raise body
                return body

        return Response()


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.c, self.o = self.root / "ComfyUI", self.root / "other"
        for rel in ("checkpoints", "diffusion_models", "text_encoders", "clip", "vae",
                    "loras"):
            (self.c / "models" / rel).mkdir(parents=True)
        (self.o / "models/illustrious").mkdir(parents=True)
        (self.o / "models/krea2").mkdir(parents=True)
        (self.o / "loras").mkdir(parents=True)
        self.paths = answer(self.c, self.o)
        self.folders = ComfyFolders(self.paths)
        self.project = self.root / "project/models"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def registry(self) -> models.Registry:
        return models.load(ModelsFolder(self.project), comfy=self.folders)

    def snapshot(self) -> list[str]:
        return sorted(str(p) for p in self.root.rglob("*"))


class FolderPathsTests(unittest.TestCase):
    def client(self, **session) -> comfy.ComfyClient:
        return comfy.ComfyClient("http://127.0.0.1:8188", session=FakeSession(**session))

    def test_answer_is_read(self) -> None:
        body = {"checkpoints": ["/a", "/b"], "loras": ["/l"]}
        self.assertEqual(self.client(body=body).folder_paths(), body)

    def test_missing_api_or_odd_answer_is_none(self) -> None:
        import requests

        for session in (
            {"status": 404, "body": {"x": ["/a"]}},
            {"body": ["/a"]},
            {"body": {"checkpoints": "/a", "loras": [1]}},
            {"body": ValueError("not json")},
            {"error": requests.ConnectionError("off")},
        ):
            with self.subTest(session=session):
                self.assertIsNone(self.client(**session).folder_paths())

    def test_odd_entries_are_dropped(self) -> None:
        body = {"checkpoints": ["/a"], "loras": "/l", "vae": [None]}
        self.assertEqual(self.client(body=body).folder_paths(), {"checkpoints": ["/a"]})


class DiscoverTests(Base):
    def test_loopback(self) -> None:
        for url in ("http://127.0.0.1:8188", "http://localhost:8188",
                    "http://LOCALHOST", "http://[::1]:8188", "http://127.0.0.2:80"):
            self.assertTrue(local.is_loopback(url), url)
        for url in ("http://192.168.0.10:8188", "https://comfy.example.com",
                    "http://0.0.0.0:8188", "not a url"):
            self.assertFalse(local.is_loopback(url), url)

    def test_remote_comfyui_is_not_asked(self) -> None:
        client = FakeComfy(self.paths)
        with self.assertLogs("jena_image_render", "INFO") as logs:
            self.assertIsNone(local.discover(client, "http://192.168.0.10:8188"))
        self.assertEqual(client.asked_paths, 0)
        self.assertIn("another computer", "\n".join(logs.output))

    def test_no_api_or_paths_elsewhere(self) -> None:
        with self.assertLogs("jena_image_render", "INFO"):
            self.assertIsNone(local.discover(FakeComfy(None), "http://127.0.0.1:8188"))
            elsewhere = {"checkpoints": ["/no/such/ComfyUI/models/checkpoints"]}
            self.assertIsNone(local.discover(FakeComfy(elsewhere), "http://127.0.0.1"))

    def test_same_computer_needs_one_existing_path(self) -> None:
        with self.assertLogs("jena_image_render", "INFO"):
            found = local.discover(FakeComfy(self.paths), "http://localhost:8188")
        self.assertIsNotNone(found)
        self.assertFalse((self.c / "models/unet").exists())  # legacy alias, absent

    def test_destination_is_the_folder_named_like_the_role(self) -> None:
        f = self.folders
        self.assertEqual(f.target("diffusion_models"), self.c / "models/diffusion_models")
        self.assertEqual(f.target("text_encoders"), self.c / "models/text_encoders")
        self.assertEqual(f.target("checkpoints"), self.c / "models/checkpoints")
        self.assertEqual(f.path("loras/zimage/x.safetensors"),
                         self.c / "models/loras/zimage/x.safetensors")
        odd = ComfyFolders({"vae": [f"{self.root}/gone", f"{self.o}/loras"],
                            "loras": [f"{self.root}/a", f"{self.root}/b"]})
        self.assertEqual(odd.target("vae"), self.o / "loras")  # first existing
        self.assertEqual(odd.target("loras"), self.root / "a")  # else the first
        with self.assertRaises(ValueError):
            odd.target("checkpoints")


class ResolutionTests(Base):
    def test_names_found_in_every_listed_folder(self) -> None:
        extra = touch(self.o / "models/illustrious/illustriousXL_v01.safetensors")
        reg = self.registry()
        model = reg.model("illustriousXL_v01")
        spec = model.files["checkpoint"]
        self.assertEqual((model.engine, spec.dest),
                         ("illustrious", "checkpoints/illustriousXL_v01.safetensors"))
        self.assertEqual(spec.local, str(extra))
        # ComfyUI lists it by its path inside that root: the leaf match copes.
        names = assets.LocalNames(FakeComfy(self.paths))
        self.assertEqual(names.find(spec), "illustriousXL_v01.safetensors")

    def test_engine_folders_are_relative_to_each_root(self) -> None:
        touch(self.o / "models/krea2/zimage/my-ft.safetensors")
        touch(self.o / "models/krea2/loose.safetensors")
        touch(self.o / "loras/zimage/z-style.safetensors")
        touch(self.c / "models/loras/any.safetensors")
        reg = self.registry()
        model = reg.model("my-ft")
        self.assertEqual(model.engine, "zimage")
        self.assertEqual(model.files["diffusion"].dest,
                         "diffusion_models/zimage/my-ft.safetensors")
        self.assertEqual(model.files["vae"], reg.model("zimage").files["vae"])
        with self.assertRaisesRegex(models.ModelsError, "engine of diffusion_models/"):
            reg.model("loose")
        self.assertEqual(reg.lora("z-style").engine, "zimage")
        self.assertEqual(reg.lora("any").engine, "")

    def test_skips_paths_that_do_not_exist(self) -> None:
        touch(self.c / "models/diffusion_models/zimage/ft.safetensors")
        self.assertEqual(self.registry().model("ft").engine, "zimage")

    def test_project_folder_wins_a_tie(self) -> None:
        mine = touch(self.project / "checkpoints/twin.safetensors")
        touch(self.c / "models/checkpoints/twin.safetensors")
        touch(self.o / "loras/style.safetensors")
        touch(self.project / "loras/style.safetensors")
        reg = self.registry()
        self.assertEqual(reg.model("twin").files["checkpoint"].local, str(mine))
        self.assertEqual(reg.lora("style").file.local,
                         str(self.project / "loras/style.safetensors"))

    def test_ambiguous_in_comfyui_and_unknown_names(self) -> None:
        touch(self.c / "models/checkpoints/a/x.safetensors")
        touch(self.o / "models/illustrious/b/x.safetensors")
        reg = self.registry()
        with self.assertRaisesRegex(models.ModelsError, "several files"):
            reg.model("x")
        with self.assertRaisesRegex(models.ModelsError, "ComfyUI's model folders"):
            reg.model("nope")

    def test_without_comfyui_folders_nothing_changes(self) -> None:
        touch(self.c / "models/checkpoints/only-there.safetensors")
        reg = models.load(ModelsFolder(self.project))
        with self.assertRaises(models.ModelsError):
            reg.model("only-there")


class FillTests(Base):
    def setUp(self) -> None:
        super().setUp()
        self.client = FakeComfy(self.paths)
        self.names = assets.LocalNames(self.client, filled=True)
        self.hashes = assets.Hashes(self.root / "sha256.json")

    def test_plan_and_fill(self) -> None:
        have = touch(self.c / "models/vae/sub/ae.safetensors")
        mine = touch(self.project / "loras/my-style.safetensors")
        files = [
            FileSpec("vae/ae.safetensors", url="https://x/ae.safetensors"),
            FileSpec("loras/my-style.safetensors", local=str(mine)),
            FileSpec(f"checkpoints/_url/{SHORT}/c.safetensors",
                     url="https://huggingface.co/o/r/resolve/main/c.safetensors",
                     size=len(DATA)),
            FileSpec("diffusion_models/zimage/z.safetensors", url="https://x/z"),
        ]
        steps, places = local.plan(files, self.names, self.folders)
        self.assertEqual([h for h, _ in steps], ["have", "link", "download", "download"])
        self.assertNotIn("vae/ae.safetensors", places)
        ckpt = self.c / f"models/checkpoints/_url/{SHORT}/c.safetensors"
        diff = self.c / "models/diffusion_models/zimage/z.safetensors"  # not unet/
        self.assertEqual(places[files[2].dest], ckpt)
        self.assertEqual(places[files[3].dest], diff)
        opener = FakeOpener()
        with self.assertLogs("jena_image_render", "INFO") as logs:
            local.fill(steps, places, self.names, self.hashes, "hf_t", opener)
        text = "\n".join(logs.output)
        self.assertIn("Downloading 2 file(s), 0.0 GB (size known for 1 of 2), "
                      "into ComfyUI's folders: ", text)
        self.assertIn(str(self.c / "models/checkpoints"), text)
        self.assertIn("Linked ", text)
        self.assertEqual((ckpt.read_bytes(), diff.read_bytes()), (DATA, DATA))
        link = self.c / "models/loras/my-style.safetensors"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), mine.resolve())
        self.assertEqual(have.read_bytes(), b"w")
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(opener.calls[0][1], {"Authorization": "Bearer hf_t"})

    def test_a_url_file_with_a_project_copy_is_linked(self) -> None:
        copy = touch(self.project / "vae/ae.safetensors")
        spec = FileSpec("vae/ae.safetensors", url="https://x/ae", local=str(copy))
        steps, _ = local.plan([spec], self.names, self.folders)
        self.assertEqual(steps[0][0], "link")

    def test_no_download_leaves_url_files_missing(self) -> None:
        mine = touch(self.project / "loras/my-style.safetensors")
        files = [FileSpec("vae/ae.safetensors", url="https://x/ae.safetensors"),
                 FileSpec("loras/my-style.safetensors", local=str(mine))]
        steps, places = local.plan(files, self.names, self.folders, download=False)
        self.assertEqual([h for h, _ in steps], ["missing", "link"])
        with self.assertLogs("jena_image_render", "INFO"), \
                self.assertRaisesRegex(assets.AssetError, "--no-download"):
            local.fill(steps, places, self.names, self.hashes, "", FakeOpener())
        self.assertTrue((self.c / "models/loras/my-style.safetensors").is_symlink())

    def test_without_folders_everything_unlisted_is_missing(self) -> None:
        mine = touch(self.project / "loras/my-style.safetensors")
        files = [FileSpec("vae/ae.safetensors", url="https://x/ae.safetensors"),
                 FileSpec("loras/my-style.safetensors", local=str(mine))]
        steps, places = local.plan(files, self.names, None)
        self.assertEqual(([h for h, _ in steps], places), (["missing", "missing"], {}))

    def test_link_falls_back_to_hard_link_then_copy(self) -> None:
        source = touch(self.root / "src/w.safetensors", b"weights")

        def refuse(*args):
            raise OSError("not allowed")

        calls = []

        def hard(src, dst):
            calls.append("hard")
            os.link(src, dst)

        with self.assertLogs("jena_image_render", "INFO") as logs:
            self.assertEqual(local.link(source, self.root / "a/w.safetensors",
                                        symlink=refuse, hardlink=hard), "hard-linked")
            self.assertEqual(local.link(source, self.root / "b/w.safetensors",
                                        symlink=refuse, hardlink=refuse), "copied")
            self.assertEqual(local.link(source, self.root / "b/w.safetensors"), "there")
        self.assertEqual(calls, ["hard"])
        self.assertEqual(os.stat(self.root / "a/w.safetensors").st_ino,
                         os.stat(source).st_ino)
        copied = self.root / "b/w.safetensors"
        self.assertFalse(copied.is_symlink())
        self.assertEqual(copied.read_bytes(), b"weights")
        self.assertFalse(copied.with_name("w.safetensors.part").exists())
        self.assertIn("Copied ", "\n".join(logs.output))


class CliLocalTests(Base):
    def setUp(self) -> None:
        super().setUp()
        self.cwd = os.getcwd()
        self.project.mkdir(parents=True)
        os.chdir(self.project.parent)
        os.environ["JENA_COMFY_URL"] = "http://127.0.0.1:8188"
        touch(self.project / "loras/my-style.safetensors")
        Path("job.json").write_text(json.dumps(
            {"version": 1, "model": ADDRESS, "loras": ["my-style"], "prompt": "x"}
        ))
        self.fake = FakeComfy(self.paths)
        self.real = cli.ComfyClient
        self.urls: list[str] = []

        def factory(url):
            self.urls.append(url)
            return self.fake

        cli.ComfyClient = factory

    def tearDown(self) -> None:
        cli.ComfyClient = self.real
        os.environ.pop("JENA_COMFY_URL", None)
        os.chdir(self.cwd)
        super().tearDown()

    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), \
                self.assertLogs("jena_image_render", "INFO") as logs:
            cli.logging.getLogger("jena_image_render").info("start")
            code = cli.main(list(argv))
        return code, out.getvalue(), "\n".join(logs.output)

    @property
    def ckpt(self) -> Path:
        return self.c / f"models/checkpoints/_url/{SHORT}/their-ckpt.safetensors"

    def test_dry_run_shows_the_plan_and_writes_nothing(self) -> None:
        touch(self.o / "models/illustrious/already.safetensors")
        before = self.snapshot()
        code, out, logs = self.run_cli("job.json", "--dry-run")
        self.assertEqual(code, 0, out + logs)
        self.assertIn("files (local ComfyUI):", out)
        self.assertIn(f"download https://huggingface.co/someone/their-model/resolve/main/"
                      f"sub/their-ckpt.safetensors → {self.ckpt}", out)
        self.assertIn("link     models/loras/my-style.safetensors → "
                      f"{self.c}/models/loras/my-style.safetensors", out)
        self.assertEqual(self.snapshot(), before)
        Path("job.json").write_text(json.dumps(
            {"version": 1, "model": "already", "prompt": "x"}))  # only in ComfyUI
        code, out, _ = self.run_cli("job.json", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("have     models/checkpoints/already.safetensors", out)

    def test_dry_run_no_download_reports_missing(self) -> None:
        code, out, _ = self.run_cli("job.json", "--dry-run", "--no-download")
        self.assertEqual(code, 1)
        self.assertIn("missing  models/checkpoints/_url/", out)
        self.assertIn("--no-download is on", out)
        self.assertIn("missing files (ComfyUI does not list them):", out)

    def test_run_downloads_links_and_renders(self) -> None:
        opener = FakeOpener()
        with mock.patch("jena_image_render.pull.requests.get", opener):
            code, _, logs = self.run_cli("job.json", "--outputs", "out")
        self.assertEqual(code, 0, logs)
        self.assertEqual(self.ckpt.read_bytes(), DATA)
        self.assertTrue((self.c / "models/loras/my-style.safetensors").is_symlink())
        graph = json.dumps(self.fake.graphs[0])
        self.assertIn(f"_url/{SHORT}/their-ckpt.safetensors", graph)
        self.assertIn('"my-style.safetensors"', graph)
        self.assertTrue(Path("out/job/their-ckpt/job1_0.png").is_file())
        self.assertNotIn("WARNING", logs)

    def test_no_download_fails_but_still_links(self) -> None:
        code, _, logs = self.run_cli("job.json", "--no-download")
        self.assertEqual(code, 1)
        self.assertIn("--no-download is on", logs)
        self.assertTrue((self.c / "models/loras/my-style.safetensors").is_symlink())
        self.assertFalse(self.ckpt.exists())
        self.assertEqual(self.fake.graphs, [])

    def test_remote_comfyui_gets_nothing_written(self) -> None:
        before = self.snapshot()
        code, _, logs = self.run_cli("job.json", "--local", "http://192.168.0.10:8188")
        self.assertEqual(code, 1)
        self.assertEqual(self.fake.asked_paths, 0)
        self.assertIn("cannot be filled from here", logs)
        self.assertIn("that computer's ComfyUI models/ folder", logs)
        self.assertEqual(self.snapshot(), before)
        code, out, _ = self.run_cli("job.json", "--dry-run", "--local",
                                    "http://192.168.0.10:8188")
        self.assertEqual(code, 1)
        self.assertIn("this ComfyUI cannot be filled from here", out)

    def test_comfyui_without_the_api_falls_back(self) -> None:
        self.fake.paths = None
        before = self.snapshot()
        code, _, logs = self.run_cli("job.json")
        self.assertEqual(code, 1)
        self.assertIn("does not report its model folders", logs)
        self.assertIn("extra_model_paths.yaml", logs)
        self.assertEqual(self.snapshot(), before)

    def test_runpod_ignores_comfyui(self) -> None:
        code, out, _ = self.run_cli("job.json", "--runpod", "--dry-run")
        self.assertEqual(code, 0)
        self.assertEqual(self.urls, [])
        self.assertIn("files (RunPod):", out)

    def test_pull_goes_into_comfyui_folders(self) -> None:
        opener = FakeOpener()
        with mock.patch("jena_image_render.pull.requests.get", opener):
            code, _, logs = self.run_cli("pull", "job.json")
        self.assertEqual(code, 0, logs)
        self.assertEqual(self.ckpt.read_bytes(), DATA)
        self.assertFalse((self.project / "checkpoints").exists())
        self.assertIn("1 file(s) downloaded into ComfyUI's model folders", logs)
        code, _, logs = self.run_cli("pull", "job.json")  # ComfyUI lists it now
        self.assertIn("ComfyUI already has checkpoints/_url/", logs)
        self.assertEqual(len(opener.calls), 1)

    def test_pull_without_comfyui_uses_the_models_folder(self) -> None:
        self.fake.up = False
        opener = FakeOpener()
        with mock.patch("jena_image_render.pull.requests.get", opener):
            code, _, logs = self.run_cli("pull", "job.json")
        self.assertEqual(code, 0, logs)
        self.assertTrue((self.project / f"checkpoints/_url/{SHORT}/"
                         "their-ckpt.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
