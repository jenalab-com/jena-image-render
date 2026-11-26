"""jenair ui: the server, started for real on a free port, with a fake ComfyUI and
a fake RunPod. No key from the environment is used, and none may be answered."""

from __future__ import annotations

import http.client
import json
import logging
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from jena_image_render import models, pod, server
from jena_image_render.comfy import ComfyError
from jena_image_render.folder import ComfyFolders, ModelsFolder

KEYS = ("RUNPOD_API_KEY", "HF_TOKEN", "CIVITAI_TOKEN", "JENA_STORAGE", "JENA_TARGET",
        "JENA_COMFY_URL", "JENA_MODELS_DIR", "JENA_POD_IMAGE")
SECRET = "fake-key-value-that-must-never-be-answered"
PROJECT = {
    "output": "t",
    "batches": [
        {"name": "a", "defaults": {"model": "ill", "seeds": [1, 2]},
         "jobs": [{"name": "x", "prompt": "a cafe"}]},
        {"name": "b", "skip": True, "jobs": [{"model": "nope", "prompt": "p"}]},
    ],
}


class FakeComfy:
    """Answers like a ComfyUI that lists ill.safetensors; counts every request."""

    def __init__(self) -> None:
        self.requests = 0
        self.paths: dict[str, list[str]] | None = None
        self.hold: threading.Event | None = None
        self.interrupted = threading.Event()
        self.graphs: list[dict] = []

    def ready(self, timeout=15):
        self.requests += 1
        return True

    def probe(self, timeout=15):
        self.requests += 1
        return "200"

    def system_stats(self, timeout=15):
        self.requests += 1
        return {"system": {"comfyui_version": "0.38.0"}, "devices": [{"name": "cuda:0 TEST"}]}

    def folder_paths(self, timeout=15):
        self.requests += 1
        return self.paths

    def options(self, folder):
        self.requests += 1
        return {"checkpoints": ["ill.safetensors"]}.get(folder, [])

    def run(self, graph, output_node, on_step=None):
        self.requests += 1
        self.graphs.append(graph)
        if on_step is not None:
            on_step(1, 2)
            on_step(2, 2)
        if self.hold is not None:
            self.hold.wait(5)
            if self.interrupted.is_set():
                raise ComfyError("interrupted")
        return b"\x89PNG fake"

    def interrupt(self):
        self.requests += 1
        self.interrupted.set()
        if self.hold is not None:
            self.hold.set()


class FakeAccount:
    def __init__(self) -> None:
        self.calls = 0
        self.pods = [{"id": "pod1", "name": "jena-render-abc"}]

    def account(self):
        self.calls += 1
        return {"clientBalance": 4.5, "currentSpendPerHr": 0.3, "pods": list(self.pods)}


class FakePods:
    def __init__(self, root: Path) -> None:
        self.api = FakeAccount()
        self.settings = pod.Settings()
        self.sessions = pod.Sessions(root / ".jena")
        self.pod_client = FakeComfy()
        self.stopped = False
        self.created = datetime.now(timezone.utc).isoformat()

    def live(self):
        return None

    def start(self, max_hours, secure=False, progress=None, should_stop=None):
        info = {"pod_id": "pod1", "gpu": "RTX TEST", "created": self.created,
                "typical_seconds": 300}
        progress({"phase": "renting", "cloud": "community", "attempt": 1})
        progress({**info, "phase": "pulling image", "elapsed": 1})
        progress({**info, "phase": "ready", "elapsed": 2})
        session = {"pod_id": "pod1", "gpu": "RTX TEST", "url": "https://pod1-3000.x",
                   "user": "render", "password": "pod-password", "created": self.created,
                   "max_hours": max_hours}
        self.sessions.save(session)
        return session

    def client(self, session):
        return self.pod_client

    def stop(self):
        self.stopped = True
        self.api.pods = []
        return True


class FakeRunPodAPI:
    def __init__(self, key):
        self.key = key

    def account(self):
        if self.key != SECRET:
            raise RuntimeError(f"401 Unauthorized for key {self.key}")
        return {"clientBalance": 1.5}


class FakeHub:
    def __init__(self, token):
        self.token = token

    def whoami(self):
        if self.token != SECRET:
            raise ValueError(f"Invalid token {self.token}\nsecond line")
        return {"name": "someone", "auth": {"accessToken": {"role": "write"}}}


class ServerBase(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {key: os.environ.pop(key, None) for key in KEYS}
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.cwd = os.getcwd()
        os.chdir(self.root)
        (self.root / "models/checkpoints").mkdir(parents=True)
        (self.root / "models/checkpoints/ill.safetensors").write_bytes(b"w")
        (self.root / "models/loras/illustrious").mkdir(parents=True)
        (self.root / "models/loras/illustrious/my-style.safetensors").write_bytes(b"l")
        (self.root / "configs/basics").mkdir(parents=True)
        (self.root / "configs/basics/one.json").write_text(json.dumps(PROJECT))
        (self.root / ".env").write_text(f"RUNPOD_API_KEY={SECRET}\n")  # never read
        self.comfy = FakeComfy()
        self.pods = FakePods(self.root)
        self.env_file = self.root / ".env"
        self.app = server.App(self.root, Path("outputs"), comfy=lambda url: self.comfy,
                              pods=lambda: self.pods, env_file=self.env_file,
                              runpod_api=FakeRunPodAPI, hf_api=FakeHub)
        self.httpd = server.make_server(self.app, 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, args=(0.05,),
                                       daemon=True)
        self.thread.start()
        self.bodies: list[bytes] = []

    def tearDown(self) -> None:
        self.app.cancel.set()
        if self.comfy.hold is not None:
            self.comfy.hold.set()
        for task in (self.app.task, self.app.stopper):
            if task is not None:
                task.join(5)
        self.httpd.shutdown()
        self.httpd.server_close()
        self.app.close()
        os.chdir(self.cwd)
        self.tmp.cleanup()
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
        for body in self.bodies:  # no answer may carry a key or the pod login
            self.assertNotIn(SECRET.encode(), body)
            self.assertNotIn(b"pod-password", body)

    def call(self, method, path, body=None, token=True, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Host": host or f"127.0.0.1:{self.port}"}
        if token:
            headers[server.TOKEN_HEADER] = self.app.token if token is True else token
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        self.bodies.append(raw)
        if "json" in (response.getheader("Content-Type") or ""):
            return response.status, json.loads(raw)
        return response.status, raw

    def wait(self, test, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, status = self.call("GET", "/api/status")
            if test(status):
                return status
            time.sleep(0.05)
        self.fail(f"status never got there: {status}")


class ServerTests(ServerBase):

    # ── safety ───────────────────────────────────────────────────────
    def test_token_is_required(self) -> None:
        self.assertEqual(self.call("GET", "/api/meta", token=False)[0], 403)
        self.assertEqual(self.call("GET", "/api/meta", token="wrong")[0], 403)
        self.assertEqual(self.call("POST", "/api/check", {"project": PROJECT},
                                   token=False)[0], 403)
        self.assertEqual(self.call("GET", "/api/meta")[0], 200)

    def test_page_carries_the_token(self) -> None:
        status, page = self.call("GET", "/", token=False)
        self.assertEqual(status, 200)
        self.assertIn(self.app.token.encode(), page)
        self.assertNotIn(server.TOKEN_SLOT.encode(), page)
        self.assertIn(b'<script id="convert">', page)

    def test_host_must_be_this_computer(self) -> None:
        for host in ("evil.example:%s" % self.port, "127.0.0.1", "localhost:1",
                     "127.0.0.1.evil.example:%s" % self.port):
            with self.subTest(host=host):
                self.assertEqual(self.call("GET", "/", host=host)[0], 403)
                self.assertEqual(self.call("GET", "/api/meta", host=host)[0], 403)
        self.assertEqual(self.call("GET", "/api/meta", host=f"localhost:{self.port}")[0],
                         200)
        self.assertEqual(self.httpd.server_address[0], "127.0.0.1")

    def test_paths_stay_inside(self) -> None:
        (self.root / "outputs").mkdir()
        (self.root / "secret.json").write_text("{}")
        for path in ("../secret.json", "/etc/passwd", ".env", "secret.json",
                     "jobs/../secret.json", "configs/../.env", "README.md"):
            with self.subTest(path=path):
                status, _ = self.call("GET", "/api/file?path=" + quote(path))
                self.assertIn(status, (400, 403))
        for path in ("../.env", "/etc/passwd", "../secret.json"):
            with self.subTest(image=path):
                status, _ = self.call("GET", "/api/image?path=" + quote(path))
                self.assertIn(status, (400, 403))
        self.assertEqual(self.call("GET", "/api/image?path=t/none.png")[0], 404)
        for name in ("../x", "a/b", ".hidden", "", "x" * 200):
            with self.subTest(name=name):
                status, _ = self.call("POST", "/api/file", {"name": name, "project": {}})
                self.assertEqual(status, 400)
        self.assertFalse((self.root / "x.json").exists())

    def test_meta_says_only_whether_keys_are_set(self) -> None:
        os.environ["RUNPOD_API_KEY"] = SECRET
        os.environ["HF_TOKEN"] = SECRET
        status, meta = self.call("GET", "/api/meta")
        self.assertEqual(status, 200)
        settings = meta["settings"]
        self.assertIs(settings["runpod_key"], True)
        self.assertIs(settings["hf_token"], True)
        self.assertEqual(settings["storage"], "")
        self.assertEqual(settings["default_target"], "local")
        self.assertEqual(settings["idle_minutes"], 20)
        engines = {e["name"]: e for e in meta["engines"]}
        self.assertEqual((engines["illustrious"]["steps"], engines["illustrious"]["cfg"]),
                         (28, 6.0))
        self.assertTrue(engines["zimage"]["cfg_fixed"])
        self.assertIn("16:9", meta["aspects"])
        self.assertIn("Euler a", meta["samplers"])
        self.assertIn("karras", meta["schedulers"])

    def test_bad_requests(self) -> None:
        self.assertEqual(self.call("GET", "/api/nothing")[0], 404)
        self.assertEqual(self.call("GET", "/elsewhere")[0], 404)
        self.assertEqual(self.call("POST", "/api/check", b"not json")[0], 400)
        self.assertEqual(self.call("POST", "/api/check", [1, 2])[0], 400)
        self.assertEqual(self.call("POST", "/api/connect", {"target": "gpu"})[0], 400)
        status, answer = self.call("POST", "/api/connect", {"target": "runpod"})
        self.assertEqual(status, 400)  # no key: nothing is rented
        self.assertIn("RUNPOD_API_KEY", answer["error"])

    # ── choices, check, files ────────────────────────────────────────
    def test_choices_resolve_with_the_same_rules(self) -> None:
        comfy = self.root / "ComfyUI/models"
        (comfy / "checkpoints/sub").mkdir(parents=True)
        (comfy / "checkpoints/sub/only-comfy.safetensors").write_bytes(b"c")
        (comfy / "checkpoints/ill.safetensors").write_bytes(b"twin")
        self.comfy.paths = {"checkpoints": [str(comfy / "checkpoints")]}
        status, found = self.call("GET", "/api/choices?target=local")
        self.assertEqual(status, 200)
        by_name = {m["name"]: m for m in found["models"]}
        self.assertEqual(by_name["zimage"]["source"], "built-in")
        self.assertEqual((by_name["ill"]["source"], by_name["ill"]["engine"]),
                         ("models folder", "illustrious"))
        self.assertEqual(by_name["only-comfy"]["source"], "ComfyUI")
        loras = {lo["name"]: lo for lo in found["loras"]}
        self.assertEqual(loras["illustrious-niji"]["trigger"], "midniji")
        self.assertEqual((loras["my-style"]["engine"], loras["my-style"]["source"]),
                         ("illustrious", "models folder"))
        registry = models.load(ModelsFolder(), comfy=ComfyFolders(self.comfy.paths))
        for item in found["models"]:
            self.assertEqual(registry.model(item["name"]).engine, item["engine"])
        for item in found["loras"]:
            self.assertEqual(registry.lora(item["name"]).engine, item["engine"])
        _, remote = self.call("GET", "/api/choices?target=runpod")
        self.assertNotIn("only-comfy", {m["name"] for m in remote["models"]})
        self.assertEqual(self.call("GET", "/api/choices?target=gpu")[0], 400)

    def test_check(self) -> None:
        status, result = self.call("POST", "/api/check", {"project": PROJECT})
        self.assertEqual(status, 200)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["images"], result["skipped"]), (2, 0))
        self.assertEqual(result["per_batch"], [{"name": "a", "images": 2, "saved": 0}])
        self.assertEqual([(f["how"], f["dest"]) for f in result["files"]],
                         [("have", "checkpoints/ill.safetensors")])
        bad = {"model": "nope", "prompt": "x"}
        status, result = self.call("POST", "/api/check", {"project": bad})
        self.assertEqual(status, 200)
        self.assertFalse(result["ok"])
        self.assertIn("'nope' is not a known alias", result["errors"][0])
        noisy = {"model": "zimage", "prompt": "x", "negative": "blur"}
        _, result = self.call("POST", "/api/check", {"project": noisy, "target": "runpod"})
        self.assertTrue(result["ok"], result)
        self.assertIn("ignores negative", "\n".join(result["warnings"]))
        self.assertEqual({f["how"] for f in result["files"]}, {"fetch"})
        _, result = self.call("POST", "/api/check", {"project": "nonsense"})
        self.assertFalse(result["ok"])

    def test_save_open_and_list(self) -> None:
        body = {"name": "my job", "project": PROJECT}
        status, saved = self.call("POST", "/api/file", body)
        self.assertEqual((status, saved["path"]), (200, "jobs/my job.json"))
        self.assertEqual(json.loads((self.root / "jobs/my job.json").read_text()), PROJECT)
        self.assertEqual(self.call("POST", "/api/file", body)[0], 409)
        self.assertEqual(self.call("POST", "/api/file", {**body, "overwrite": True})[0], 200)
        self.assertEqual(self.call("POST", "/api/file",
                                   {"name": "again.json", "project": {}})[1]["path"],
                         "jobs/again.json")
        _, files = self.call("GET", "/api/files")
        self.assertEqual(files, {"jobs": ["jobs/again.json", "jobs/my job.json"],
                                 "examples": ["configs/basics/one.json"]})
        status, opened = self.call("GET", "/api/file?path=" + quote("jobs/my job.json"))
        self.assertEqual((status, opened["project"], opened["name"]), (200, PROJECT, "my job"))
        _, example = self.call("GET", "/api/file?path=configs/basics/one.json")
        self.assertEqual(example["project"], PROJECT)
        (self.root / "jobs/broken.json").write_text("{")
        self.assertEqual(self.call("GET", "/api/file?path=jobs/broken.json")[0], 400)

    # ── connect, render, cancel ──────────────────────────────────────
    def test_connect_local(self) -> None:
        self.assertEqual(self.call("POST", "/api/connect", {"target": "local"})[0], 200)
        status = self.wait(lambda s: s["connection"]["state"] == "ready")
        self.assertEqual(status["target"], "local")
        detail = status["connection"]["detail"]
        self.assertEqual((detail["version"], detail["device"], detail["same_computer"]),
                         ("0.38.0", "cuda:0 TEST", False))
        self.call("POST", "/api/disconnect", {})
        self.wait(lambda s: s["connection"]["state"] == "idle")

    def test_render_fills_results(self) -> None:
        logger = logging.getLogger("jena_image_render")
        level = logger.level
        logger.setLevel(logging.INFO)  # as under `jenair ui`
        self.addCleanup(logger.setLevel, level)
        status, started = self.call("POST", "/api/render",
                                    {"project": PROJECT, "target": "local"})
        self.assertEqual(status, 200)
        self.assertTrue(started["started"], started)
        final = self.wait(lambda s: s["render"]["state"] in ("done", "error"))
        render = final["render"]
        self.assertEqual(render["state"], "done", final)
        self.assertEqual((render["total"], render["done"], render["failed"]), (2, 2, 0))
        self.assertEqual([(r["batch"], r["model"], r["name"], r["seed"], r["ok"])
                          for r in render["results"]],
                         [("a", "ill", "x", 1, True), ("a", "ill", "x", 2, True)])
        self.assertEqual(render["results"][0]["path"], "t/a/ill/x_1.png")
        status, image = self.call("GET", "/api/image?path=t/a/ill/x_1.png")
        self.assertEqual((status, image), (200, b"\x89PNG fake"))
        self.assertTrue((self.root / "outputs/t/a/ill/x_2.json").is_file())
        self.assertEqual(final["files"]["state"], "done")
        self.assertTrue(any("2 saved, 0 failed" in line for line in final["log"]),
                        final["log"])
        _, again = self.call("POST", "/api/render", {"project": PROJECT, "target": "local"})
        self.assertFalse(again["started"])  # everything is saved now
        self.assertEqual(again["skipped"], 2)

    def test_render_needs_a_target_and_a_valid_project(self) -> None:
        self.assertEqual(self.call("POST", "/api/render", {"project": PROJECT})[0], 400)
        _, result = self.call("POST", "/api/render",
                              {"project": {"model": "nope", "prompt": "x"},
                               "target": "local"})
        self.assertFalse(result["started"])
        self.assertFalse(result["ok"])

    def test_cancel_stops_after_the_current_image(self) -> None:
        self.comfy.hold = threading.Event()
        self.call("POST", "/api/render", {"project": PROJECT, "target": "local"})
        self.wait(lambda s: (s["render"]["current"] or {}).get("step") == 2)
        status, _ = self.call("POST", "/api/render", {"project": PROJECT, "target": "local"})
        self.assertEqual(status, 409)  # one render at a time
        self.assertEqual(self.call("POST", "/api/cancel", {})[0], 200)
        final = self.wait(lambda s: s["render"]["state"] != "running")
        self.assertEqual(final["render"]["state"], "cancelled")
        self.assertEqual((final["render"]["done"], final["render"]["failed"]), (0, 0))
        self.assertTrue(self.comfy.interrupted.is_set())
        self.assertEqual(len(self.comfy.graphs), 1)

    def test_runpod_status_never_asks_the_pod(self) -> None:
        os.environ["RUNPOD_API_KEY"] = SECRET
        self.call("POST", "/api/connect", {"target": "runpod", "max_hours": 2})
        status = self.wait(lambda s: s["connection"]["state"] == "ready")
        asked = self.pods.pod_client.requests
        calls = self.pods.api.calls
        for _ in range(5):
            _, status = self.call("GET", "/api/status")
        self.assertEqual(self.pods.pod_client.requests, asked)
        self.assertLessEqual(self.pods.api.calls, calls + 1)  # cached for 10 seconds
        pod_info = status["pod"]
        self.assertEqual((pod_info["id"], pod_info["gpu"], pod_info["phase"]),
                         ("pod1", "RTX TEST", "ready"))
        self.assertEqual((pod_info["balance"], pod_info["spend_per_hour"]), (4.5, 0.3))
        now = time.time()
        self.assertAlmostEqual(pod_info["idle_deadline"], now + 20 * 60, delta=30)
        created = datetime.fromisoformat(self.pods.created).timestamp()
        self.assertAlmostEqual(pod_info["hard_deadline"], created + 2 * 3600, delta=1)
        self.call("POST", "/api/disconnect", {})
        final = self.wait(lambda s: s["connection"]["state"] == "idle" and not s["busy"])
        self.assertTrue(self.pods.stopped)
        self.assertEqual(final["pod"]["id"], "")


if __name__ == "__main__":
    unittest.main()


class SettingsTests(ServerBase):
    """The Settings screen: .env written in place, secrets never answered."""

    def setUp(self) -> None:
        super().setUp()
        self.env_file.write_text("# my comment\nOTHER=1\nRUNPOD_API_KEY=old\n\n"
                                 "export JENA_TARGET=local\n")

    def save(self, values):
        return self.call("POST", "/api/settings", {"values": values})

    def test_read_write_and_clear(self) -> None:
        status, read = self.call("GET", "/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(read["secrets"]["RUNPOD_API_KEY"],
                         {"set": False, "from_environment": False})
        status, saved = self.save({"RUNPOD_API_KEY": f"  {SECRET} ", "HF_TOKEN": SECRET,
                                   "JENA_TARGET": "runpod",
                                   "JENA_STORAGE": "hf://me/models"})
        self.assertEqual(status, 200)
        self.assertTrue(saved["secrets"]["RUNPOD_API_KEY"]["set"])
        self.assertEqual(saved["values"]["JENA_STORAGE"]["value"], "hf://me/models")
        self.assertEqual(self.env_file.read_text().splitlines(), [
            "# my comment", "OTHER=1", f"RUNPOD_API_KEY={SECRET}", "",
            "JENA_TARGET=runpod", f"HF_TOKEN={SECRET}", "JENA_STORAGE=hf://me/models"])
        self.assertEqual(self.env_file.stat().st_mode & 0o777, 0o600)
        self.assertEqual(os.environ["RUNPOD_API_KEY"], SECRET)  # no restart needed
        self.assertTrue(self.call("GET", "/api/meta")[1]["settings"]["runpod_key"])
        self.save({"RUNPOD_API_KEY": None, "JENA_STORAGE": ""})
        lines = self.env_file.read_text().splitlines()
        self.assertIn("RUNPOD_API_KEY=", lines)
        self.assertIn("JENA_STORAGE=", lines)
        self.assertNotIn("RUNPOD_API_KEY", os.environ)
        self.assertFalse(self.call("GET", "/api/settings")[1]["secrets"]
                         ["RUNPOD_API_KEY"]["set"])

    def test_refused_values_change_nothing(self) -> None:
        before = self.env_file.read_text()
        for values in ({"PATH": "/bin"}, {"RUNPOD_API_KEY": f"{SECRET}\nPATH=/x"},
                       {"HF_TOKEN": f"a\0{SECRET}"}, {"JENA_TARGET": "gpu"},
                       {"JENA_STORAGE": "s3://x"}, {"HF_TOKEN": 5}):
            with self.subTest(values=list(values)):
                self.assertEqual(self.save(values)[0], 400)
        self.assertEqual(self.call("POST", "/api/settings", {"values": [1]})[0], 400)
        self.assertEqual(self.env_file.read_text(), before)

    def test_a_failing_write_says_so_without_the_value(self) -> None:
        self.app.env_file = self.root / "a-folder"
        self.app.env_file.mkdir()
        status, answer = self.save({"RUNPOD_API_KEY": SECRET})
        self.assertEqual(status, 500)
        self.assertIn("could not write", answer["error"])
        self.assertNotIn("RUNPOD_API_KEY", os.environ)

    def test_cached_clients_are_dropped(self) -> None:
        self.app.lookup = object()
        self.app.pods = object()
        self.save({"RUNPOD_API_KEY": SECRET})
        self.assertIsNone(self.app.lookup)
        self.assertIsNone(self.app.pods)  # no session: made again on connect
        self.call("POST", "/api/connect", {"target": "runpod"})
        self.wait(lambda s: s["connection"]["state"] == "ready")  # works at once
        self.save({"JENA_COMFY_URL": "http://127.0.0.1:8190"})
        self.assertEqual(self.app.comfy_url, "http://127.0.0.1:8190")

    def test_the_environment_wins(self) -> None:
        os.environ["HF_TOKEN"] = "from-the-shell"
        app = server.App(self.root, env_file=self.env_file)
        self.addCleanup(app.close)
        self.assertTrue(app.settings()["secrets"]["HF_TOKEN"]["from_environment"])
        self.assertFalse(app.settings()["secrets"]["RUNPOD_API_KEY"]["from_environment"])
        app.save_settings({"values": {"HF_TOKEN": SECRET}})
        self.assertEqual(os.environ["HF_TOKEN"], "from-the-shell")
        self.assertIn(f"HF_TOKEN={SECRET}", self.env_file.read_text())

    def test_key_tests(self) -> None:
        status, answer = self.call("POST", "/api/settings/test", {"which": "runpod"})
        self.assertEqual((status, answer["ok"]), (200, False))
        self.save({"RUNPOD_API_KEY": SECRET, "HF_TOKEN": SECRET})
        _, answer = self.call("POST", "/api/settings/test", {"which": "runpod"})
        self.assertEqual(answer, {"ok": True, "message": "RunPod account balance: $1.50"})
        _, answer = self.call("POST", "/api/settings/test", {"which": "hf"})
        self.assertEqual(answer["message"], "Hugging Face account: someone (write token)")
        wrong = "wrong-" + SECRET[::-1]
        for which in ("runpod", "hf"):
            _, answer = self.call("POST", "/api/settings/test",
                                  {"which": which, "value": wrong})
            self.assertFalse(answer["ok"])
            self.assertIn("***", answer["message"])
            self.assertNotIn(wrong, answer["message"])
            self.assertNotIn("\n", answer["message"])
        self.assertEqual(self.call("POST", "/api/settings/test", {"which": "x"})[0], 400)
        self.assertFalse(any(wrong.encode() in body for body in self.bodies))
        self.assertFalse(any(SECRET in line for line in self.app.log))


def refuse(*args):
    raise OSError("not allowed")


class BrowseTests(ServerBase):
    def setUp(self) -> None:
        super().setUp()
        self.pick = self.root / "pick"
        (self.pick / "sub").mkdir(parents=True)
        (self.pick / ".hidden").mkdir()
        for name, data in (("a.safetensors", b"aaaa"), ("B.GGUF", b"b"),
                           (".h.safetensors", b"h"), ("notes.txt", b"n"),
                           ("old.pt", b"p")):
            (self.pick / name).write_bytes(data)

    def browse(self, path, kind="loras"):
        return self.call("GET", f"/api/browse?kind={kind}&path=" + quote(str(path)))

    def bring(self, path, kind="loras", **extra):
        return self.call("POST", "/api/import", {"path": str(path), "kind": kind, **extra})

    def test_listing(self) -> None:
        status, found = self.browse(self.pick)
        self.assertEqual(status, 200)
        self.assertTrue(found["ok"])
        self.assertEqual((found["path"], found["parent"]), (str(self.pick), str(self.root)))
        self.assertEqual(found["dirs"], ["sub"])
        self.assertEqual(found["files"], [{"name": "a.safetensors", "size": 4},
                                          {"name": "B.GGUF", "size": 1}])
        names = {p["name"]: p["path"] for p in found["places"]}
        self.assertEqual(names["models/loras"], str(self.root / "models/loras"))
        self.assertIn("Home", names)
        _, models_root = self.browse(self.pick, kind="diffusion_models")
        self.assertIn(str(self.root / "models"), [p["path"] for p in models_root["places"]])
        _, start = self.browse("")
        self.assertEqual(start["path"], found["places"][0]["path"])
        self.assertEqual(self.browse("relative/path")[0], 400)
        self.assertEqual(self.browse(self.pick, kind="vae")[0], 400)
        _, gone = self.browse(self.root / "nowhere")
        self.assertFalse(gone["ok"])
        self.assertEqual(self.browse(self.pick, kind="loras")[0], 200)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root reads all")
    def test_unreadable_folder_is_a_message(self) -> None:
        locked = self.root / "locked"
        locked.mkdir()
        locked.chmod(0)
        try:
            status, found = self.browse(locked)
        finally:
            locked.chmod(0o700)
        self.assertEqual(status, 200)
        self.assertFalse(found["ok"])
        self.assertIn("cannot be read", found["message"])

    def resolves(self, answer, kind="loras"):
        registry = models.load(ModelsFolder())
        if kind == "loras":
            return registry.lora(answer["name"]).file.dest
        return registry.model(answer["name"]).engine

    def test_import_a_lora(self) -> None:
        status, answer = self.bring(self.pick / "a.safetensors", engine="klein")
        self.assertEqual(status, 200, answer)
        self.assertEqual((answer["name"], answer["engine"], answer["how"]),
                         ("a", "klein", "linked"))
        dest = self.root / "models/loras/klein/a.safetensors"
        self.assertTrue(dest.is_symlink())
        self.assertEqual(self.resolves(answer), "loras/klein/a.safetensors")
        _, again = self.bring(self.pick / "a.safetensors", engine="klein")
        self.assertEqual(again["how"], "already there")
        _, any_engine = self.bring(self.pick / "B.GGUF")
        self.assertEqual((any_engine["name"], any_engine["engine"]), ("B", ""))
        self.assertEqual(self.resolves(any_engine), "loras/B.GGUF")

    def test_a_clashing_name_still_resolves(self) -> None:
        twin = self.pick / "sub/my-style.safetensors"  # models/ has one for illustrious
        twin.write_bytes(b"other")
        _, answer = self.bring(twin, engine="klein")
        self.assertEqual(answer["name"], "klein/my-style")
        self.assertEqual(self.resolves(answer), "loras/klein/my-style.safetensors")

    def test_fallbacks_hard_link_then_copy(self) -> None:
        from functools import partial

        from jena_image_render import local

        self.app.link = partial(local.link, symlink=refuse)
        _, answer = self.bring(self.pick / "a.safetensors", kind="checkpoints")
        self.assertEqual((answer["how"], answer["name"], answer["engine"]),
                         ("hard-linked", "a", "illustrious"))
        self.app.link = partial(local.link, symlink=refuse, hardlink=refuse)
        _, answer = self.bring(self.pick / "B.GGUF", kind="checkpoints")
        self.assertEqual(answer["how"], "copied")
        copied = self.root / "models/checkpoints/B.GGUF"
        self.assertFalse(copied.is_symlink())
        self.assertEqual(self.resolves(answer, "checkpoints"), "illustrious")

    def test_diffusion_models_need_a_three_file_engine(self) -> None:
        path = self.pick / "a.safetensors"
        self.assertEqual(self.bring(path, kind="diffusion_models")[0], 400)
        self.assertEqual(self.bring(path, kind="diffusion_models",
                                    engine="illustrious")[0], 400)
        _, answer = self.bring(path, kind="diffusion_models", engine="zimage")
        self.assertEqual((answer["name"], answer["engine"]), ("a", "zimage"))
        self.assertTrue((self.root / "models/diffusion_models/zimage/a.safetensors")
                        .is_symlink())
        self.assertEqual(self.resolves(answer, "diffusion_models"), "zimage")

    def test_already_inside_the_models_folder(self) -> None:
        before = sorted(str(p) for p in (self.root / "models").rglob("*"))
        _, answer = self.bring(self.root / "models/checkpoints/ill.safetensors",
                               kind="checkpoints")
        self.assertEqual((answer["name"], answer["how"]), ("ill", "already there"))
        _, lora = self.bring(self.root / "models/loras/illustrious/my-style.safetensors")
        self.assertEqual((lora["name"], lora["engine"]), ("my-style", "illustrious"))
        self.assertEqual(sorted(str(p) for p in (self.root / "models").rglob("*")),
                         before)

    def test_name_clash_needs_replace(self) -> None:
        taken = self.root / "models/checkpoints/a.safetensors"
        taken.write_bytes(b"something else entirely")
        status, answer = self.bring(self.pick / "a.safetensors", kind="checkpoints")
        self.assertEqual(status, 409)
        self.assertIn("replace", answer["error"])
        _, answer = self.bring(self.pick / "a.safetensors", kind="checkpoints",
                               replace=True)
        self.assertEqual(answer["how"], "linked")
        self.assertEqual(taken.read_bytes(), b"aaaa")

    def test_only_weight_files(self) -> None:
        self.assertEqual(self.bring(self.pick / "notes.txt")[0], 400)
        self.assertEqual(self.bring(self.pick / "missing.safetensors")[0], 404)
        self.assertEqual(self.bring("relative.safetensors")[0], 400)
        self.assertEqual(self.bring(self.pick / "a.safetensors", engine="pony")[0], 400)
