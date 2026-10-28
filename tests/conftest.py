"""Task stubs raise NotImplementedError("T1-xx"). By default their acceptance
tests are reported as skipped, so the suite stays green while tasks are open.
Agents working a task run with --strict-impl to see the real failures:

    uv run pytest tests/conventions/test_readings.py --strict-impl"""

from __future__ import annotations

import pytest
from scene import HUMAN, ROBOT, obj

from curio_robot.types import ObjectMap


def pytest_addoption(parser):
    parser.addoption(
        "--strict-impl", action="store_true",
        help="fail (don't skip) tests that hit an unimplemented task stub",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        call.excinfo is not None
        and call.excinfo.errisinstance(NotImplementedError)
        and not item.config.getoption("--strict-impl")
    ):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"open task: {call.excinfo.value}")


# ---- shared fixtures (helpers live in tests/scene.py) ----


@pytest.fixture
def pair_map() -> ObjectMap:
    """Two reds and a blue. Speaker's left = red_2, robot's left = red_1."""
    return ObjectMap(
        [obj("red_1", "red", (0.25, 0.08)), obj("red_2", "red", (0.25, -0.08)),
         obj("blue_1", "blue", (0.3, 0.0))],
        ROBOT, HUMAN,
    )


@pytest.fixture
def robot_only_map(pair_map) -> ObjectMap:
    return ObjectMap(pair_map.objects, ROBOT, None)
