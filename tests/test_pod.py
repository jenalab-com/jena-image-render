import json
import tempfile
import unittest
from pathlib import Path

from jena_image_render import pod
from jena_image_render.runpod_api import NoCapacity


class FakeAPI:
    def __init__(self, create_results) -> None:
        self.create_results = list(create_results)
        self.created: list[dict] = []
        self.terminated: list[str] = []
        self.pods: list[dict] = []

    def create_pod(self, name, image, env, gpus, secure, host_memory_gb, cuda_versions, disk_gb):
        self.created.append({"name": name, "image": image, "env": env, "secure": secure})
        result = self.create_results.pop(0)
        if isinstance(result, Exception):
            raise result
        self.pods.append({"id": result, "name": name, "desiredStatus": "RUNNING"})
        return {"id": result, "gpu": "RTX TEST"}

    def get_pod(self, pod_id):
        return next((p for p in self.pods if p["id"] == pod_id), None)

    def terminate(self, pod_id):
        self.terminated.append(pod_id)
        self.pods = [p for p in self.pods if p["id"] != pod_id]

    def account(self):
        return {"pods": list(self.pods), "clientBalance": 5, "currentSpendPerHr": 0}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class PodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.sessions = pod.Sessions(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def pods(self, api, ready):
        clock = Clock()
        pods = pod.Pods(api, pod.Settings(), self.sessions, image="img:test",
                        sleep=clock.sleep, clock=clock)
        pods.client = lambda session: type("C", (), {"ready": lambda s: ready(session), "probe": lambda s: "502"})()
        return pods

    def test_community_no_stock_falls_back_to_secure(self) -> None:
        api = FakeAPI([NoCapacity("none"), "pod2"])
        session = self.pods(api, ready=lambda s: True).start(max_hours=1)
        self.assertEqual(session["pod_id"], "pod2")
        self.assertEqual([c["secure"] for c in api.created], [False, True])
        env = api.created[1]["env"]
        self.assertEqual(env["COMFY_USER"], "render")
        self.assertEqual(env["IDLE_SECONDS"], "1200")
        self.assertEqual(env["MAX_SECONDS"], "3600")
        saved = json.loads(self.sessions.path.read_text())
        self.assertEqual(saved["password"], env["COMFY_PASSWORD"])
        self.assertEqual(self.sessions.path.stat().st_mode & 0o777, 0o600)

    def test_stuck_pull_terminates_and_retries(self) -> None:
        api = FakeAPI(["stuck", "good", "good2"])
        api.get_pod = lambda pod_id: {"desiredStatus": "RUNNING", "runtime": None} if pod_id == "stuck" else {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 5}}
        session = self.pods(api, ready=lambda s: s["pod_id"] == "good").start(max_hours=1)
        self.assertEqual(session["pod_id"], "good")
        self.assertEqual(api.terminated, ["stuck"])

    def test_ctrl_c_during_start_terminates(self) -> None:
        api = FakeAPI(["p1"])

        def ready(session):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.pods(api, ready=ready).start(max_hours=1)
        self.assertEqual(api.terminated, ["p1"])
        self.assertIsNone(self.sessions.load())

    def test_live_forgets_gone_pod_and_stop_cleans_up(self) -> None:
        api = FakeAPI([])
        self.sessions.save({"pod_id": "old", "url": "u", "user": "render", "password": "p"})
        pods = self.pods(api, ready=lambda s: True)
        self.assertIsNone(pods.live())
        self.assertIsNone(self.sessions.load())
        api.pods = [{"id": "x", "name": "jena-render-abc"}, {"id": "y", "name": "someone-else"}]
        self.assertTrue(pods.stop())
        self.assertEqual(api.terminated, ["x"])

    def test_status_hides_password(self) -> None:
        api = FakeAPI([])
        api.pods = [{"id": "x", "name": "jena-render-1"}]
        self.sessions.save({"pod_id": "x", "url": "u", "user": "render", "password": "secret"})
        status = self.pods(api, ready=lambda s: True).status()
        self.assertNotIn("password", status["session"])

    def test_settings_file(self) -> None:
        path = Path(self.tmp.name) / "runpod.json"
        path.write_text(json.dumps({"community": ["NVIDIA L4"], "idle_minutes": 5}))
        settings = pod.Settings.load(path)
        self.assertEqual((settings.community, settings.idle_minutes), (["NVIDIA L4"], 5))
        path.write_text(json.dumps({"gpus": []}))
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            pod.Settings.load(path)

    def test_phases_are_reported(self) -> None:
        api = FakeAPI(["p1"])
        api.get_pod = lambda pod_id: {"desiredStatus": "RUNNING",
                                      "runtime": {"uptimeInSeconds": 3}}
        answers = iter([False, False, True])
        events = []
        session = self.pods(api, ready=lambda s: next(answers)).start(
            max_hours=2, progress=events.append)
        phases = [e["phase"] for e in events]
        self.assertEqual(phases[0], "renting")
        self.assertEqual(events[0]["cloud"], "community")
        self.assertEqual(phases[-1], "ready")
        self.assertIn("starting ComfyUI", phases)
        self.assertEqual(events[-1]["pod_id"], "p1")
        self.assertEqual(events[-1]["typical_seconds"], pod.TYPICAL_START_SECONDS)
        self.assertEqual(session["max_hours"], 2)

    def test_should_stop_terminates_the_starting_pod(self) -> None:
        api = FakeAPI(["p1"])
        asked = []

        def stop() -> bool:
            asked.append(1)
            return len(asked) > 2

        with self.assertRaises(pod.Cancelled):
            self.pods(api, ready=lambda s: False).start(max_hours=1, should_stop=stop)
        self.assertEqual(api.terminated, ["p1"])
        self.assertEqual(len(api.created), 1)  # no Secure attempt after a cancel
        self.assertIsNone(self.sessions.load())


if __name__ == "__main__":
    unittest.main()
