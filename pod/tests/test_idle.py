"""The pod shuts itself down after idling, and never while work is running."""

import unittest

from jena_fetch.idle import IdleWatch, run_watch


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class IdleWatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.watch = IdleWatch(idle_seconds=1200, max_seconds=3 * 3600, clock=self.clock)

    def test_idle_after_quiet_period(self) -> None:
        self.clock.now = 1199
        self.assertIsNone(self.watch.reason(busy=False))
        self.clock.now = 1201
        self.assertIn("idle", self.watch.reason(busy=False) or "")

    def test_activity_resets_idle_timer(self) -> None:
        self.clock.now = 1000
        self.watch.touch()
        self.clock.now = 2100
        self.assertIsNone(self.watch.reason(busy=False))

    def test_busy_work_is_never_idle(self) -> None:
        self.clock.now = 5000
        self.assertIsNone(self.watch.reason(busy=True))
        self.clock.now = 5000 + 1199
        self.assertIsNone(self.watch.reason(busy=False))

    def test_hard_limit_applies_even_when_busy(self) -> None:
        self.clock.now = 3 * 3600 + 1
        self.assertIn("max", self.watch.reason(busy=True) or "")

    def test_disabled_limits(self) -> None:
        watch = IdleWatch(idle_seconds=0, max_seconds=0, clock=self.clock)
        self.clock.now = 10**6
        self.assertIsNone(watch.reason(busy=False))


def advancing():
    """A clock that moves 10s per reading; StopIteration ends a runaway loop."""
    ticks = iter(range(0, 10_000, 10))
    return lambda: float(next(ticks))


class RunWatchTests(unittest.TestCase):
    """The watcher must keep retrying until termination succeeds."""

    def test_retries_failed_termination_until_success(self) -> None:
        watch = IdleWatch(idle_seconds=1, max_seconds=0, clock=advancing())
        results = iter([False, False, True])
        attempts: list[str] = []

        def terminate(reason: str) -> bool:
            attempts.append(reason)
            return next(results)

        run_watch(watch, busy=lambda: False, terminate=terminate, sleep=lambda s: None)
        self.assertEqual(len(attempts), 3)

    def test_errors_in_checks_do_not_stop_the_watcher(self) -> None:
        watch = IdleWatch(idle_seconds=1, max_seconds=0, clock=advancing())
        calls = {"busy": 0}

        def busy() -> bool:
            calls["busy"] += 1
            if calls["busy"] == 1:
                raise RuntimeError("queue api changed")
            return False

        def terminate(reason: str) -> bool:
            if calls["busy"] < 2:
                raise OSError("network down")
            return True

        run_watch(watch, busy=busy, terminate=terminate, sleep=lambda s: None)
        self.assertGreaterEqual(calls["busy"], 2)


if __name__ == "__main__":
    unittest.main()
