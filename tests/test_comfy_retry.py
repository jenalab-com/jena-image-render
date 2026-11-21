"""Read-only calls ride out RunPod proxy hiccups; real answers are not retried."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import requests

from jena_image_render.comfy import GET_ATTEMPTS, ComfyClient, ComfyError
from jena_image_render.models import FileSpec


def reply(status: int, data: object = None, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(status_code=status, text=text, json=lambda: data, content=b"")


class Session:
    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self.auth = None

    def _next(self, method: str, url: str):
        self.calls.append((method, url))
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        return self._next("GET", url)

    def post(self, url, **kwargs):
        return self._next("POST", url)


class RetryTests(unittest.TestCase):
    def client(self, replies: list) -> tuple[ComfyClient, Session, list[float]]:
        session, naps = Session(replies), []
        return ComfyClient("http://pod", session=session, sleep=naps.append), session, naps

    def test_empty_404_right_after_start_is_asked_again(self) -> None:
        client, session, naps = self.client([reply(404), reply(502), reply(200, {"a": 1})])
        self.assertEqual(client.fetch_status(), {"a": 1})
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(naps), 2)

    def test_a_real_404_is_an_answer(self) -> None:
        client, session, naps = self.client([reply(404, text="404: Not Found")])
        with self.assertRaisesRegex(ComfyError, "404"):
            client.fetch_status()
        self.assertEqual((len(session.calls), naps), (1, []))

    def test_it_gives_up_after_a_few_tries(self) -> None:
        client, session, _ = self.client([reply(503)] * GET_ATTEMPTS)
        with self.assertRaisesRegex(ComfyError, "503"):
            client.fetch_status()
        self.assertEqual(len(session.calls), GET_ATTEMPTS)

    def test_a_dropped_connection_is_asked_again(self) -> None:
        client, session, _ = self.client(
            [requests.ConnectionError("reset"), reply(200, {"x": {"status": "done"}})]
        )
        self.assertEqual(client.fetch_status(), {"x": {"status": "done"}})

    def test_submitting_downloads_is_retried_too(self) -> None:
        client, session, _ = self.client([reply(404), reply(200, {"accepted": 1})])
        client.fetch([FileSpec("vae/a.safetensors", url="https://x/a", sha256="a" * 64, size=1)])
        self.assertEqual([m for m, _ in session.calls], ["POST", "POST"])

    def test_queueing_a_render_is_never_retried(self) -> None:
        client, session, _ = self.client([reply(502, text="bad gateway")])
        with self.assertRaises(ComfyError):
            client.queue({})
        self.assertEqual(len(session.calls), 1)


if __name__ == "__main__":
    unittest.main()
