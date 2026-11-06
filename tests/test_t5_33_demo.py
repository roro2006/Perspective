"""T5-33 acceptance: the demo session and server on CPU with a small world."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from curio_robot.demo import status
from curio_robot.demo.server import Hub, make_handler, parse_job
from curio_robot.demo.session import Busy, DemoSession, oracle_xy, verdict
from curio_robot.sim.scenarios import SCENARIOS
from curio_robot.sim.world import SimWorld

PANE = (480, 360)  # the demo's resolution; smaller renders can hide a block from the detector


@pytest.fixture
def session(tmp_path):
    events: list[dict] = []
    s = DemoSession(events.append, pane=PANE, record_dir=tmp_path, fps=200.0, speed=2.0)
    s.start()
    assert s.error is None, s.error
    s.events = events
    yield s
    s.close()


def _run(s: DemoSession, job: str, **args) -> None:
    s.submit(job, **args)
    assert s.wait_idle(120)


# ---- ground truth -----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(SCENARIOS))
def test_oracle_matches_the_scenarios_answer(kind):
    with SimWorld(160, 120) as w:
        for seed in range(8):
            sc = SCENARIOS[kind](w, np.random.default_rng(seed))
            xy, _ = oracle_xy(sc.request, w.block_positions(), sc.human_eye, sc.human_look)
            assert xy is not None and np.allclose(xy, sc.expected_xy, atol=1e-6)


def test_oracle_your_left_is_the_robots_left_and_unknown_is_unknown():
    blocks = [("red", np.array([0.25, 0.08, 0.0125])), ("red", np.array([0.25, -0.08, 0.0125]))]
    eye, look = np.array([0.72, 0.0, 0.38]), np.array([0.24, 0.0, 0.0])
    mine, _ = oracle_xy("the red one on my left", blocks, eye, look)
    yours, _ = oracle_xy("the red one on your left", blocks, eye, look)
    assert mine[1] < 0 < yours[1]  # person faces -x: their left is -y; the robot's is +y
    none, why = oracle_xy("the red one", blocks, eye, look)
    assert none is None and "no relation" in why
    assert verdict(None, None, why, known=False) == "unknown"
    missing, why = oracle_xy("the purple one", blocks, eye, look)  # no colour: any block
    assert missing is None


# ---- session ----------------------------------------------------------------------------------


def test_curio_flips_the_perspective_scene(session):
    _run(session, "load", kind="perspective", seed=0)
    text = session.scene["request"]
    _run(session, "compare", text=text)
    off, on = session.history[-2:]
    assert (off["curio"], on["curio"]) == (False, True)
    assert off["verdict"] == "wrong" and "robot's" in off["reason"]
    assert on["verdict"] == "correct" and "speaker's" in on["reason"]
    assert on["motion"]["status"] == "arrived" and on["motion"]["error_to_oracle_m"] < 0.02
    kinds = {e["type"] for e in session.events}
    assert {"frame", "interpretation", "result", "state"} <= kinds
    assert set(session.last_frame["panes"]) == {"robot", "head", "twin"}


def test_occluded_block_is_found_only_with_curio(session):
    _run(session, "load", kind="occlusion", seed=1)
    _run(session, "compare", text=session.scene["request"])
    off, on = session.history[-2:]
    assert off["verdict"] == "not_found" and off["motion"]["status"] == "no_target"
    assert on["verdict"] == "correct"


def test_busy_is_refused_and_cancel_stops_motion(tmp_path):
    s = DemoSession(lambda e: None, pane=PANE, record_dir=tmp_path, fps=20.0, speed=0.05)
    s.start()
    try:
        _run(s, "load", kind="perspective", seed=0)
        s.submit("run", text=s.scene["request"], curio=True)
        with pytest.raises(Busy):
            s.submit("run", text="the red one", curio=True)
        time.sleep(0.5)
        s.cancel()
        assert s.wait_idle(30)
        assert s.history[-1]["motion"]["status"] == "cancelled"
    finally:
        s.close()


def test_replay_is_labelled_and_comes_from_the_recording(session):
    _run(session, "load", kind="perspective", seed=0)
    _run(session, "run", text=session.scene["request"], curio=True)
    recs = session.recordings()
    assert len(recs) == 1 and recs[0]["verdict"] == "correct"
    n = len(session.events)
    _run(session, "replay", id=recs[0]["id"])
    replayed = [e for e in session.events[n:] if e["type"] in ("frame", "interpretation")]
    assert replayed and all(e["replay"] for e in replayed)
    assert len(session.history) == 1  # a replay is not a run


def test_bad_jobs_report_errors_not_results(session):
    _run(session, "replay", id="nope")
    assert any(e["type"] == "error" for e in session.events)


# ---- server -----------------------------------------------------------------------------------


def test_parse_job_validates():
    assert parse_job("/api/load", {"kind": "occlusion", "seed": "4"}) == (
        "load", {"kind": "occlusion", "seed": 4})
    assert parse_job("/api/run", {"text": " hi ", "curio": 0}) == (
        "run", {"text": "hi", "curio": False})
    for path, body in (("/api/load", {"kind": "x"}), ("/api/run", {"text": ""}),
                       ("/api/ab", {"trials": 10_000}), ("/api/replay", {"id": "../x"})):
        with pytest.raises(ValueError):
            parse_job(path, body)
    with pytest.raises(LookupError):
        parse_job("/api/shell", {})


def test_server_routes(session):
    hub = Hub()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session, hub))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Curio Live Console" in page
        st = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert st["preflight"] and st["dropins"] and "perspective" in st["scenes"]
        req = urllib.request.Request(base + "/api/load", json.dumps({"kind": "bad"}).encode(),
                                     {"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 400
    finally:
        httpd.shutdown()


def test_dropins_read_the_board():
    rows = {d["card"]: d for d in status.dropins()}
    assert rows["T5-32"]["status"] == "done" and "FAIL" in rows["T5-32"]["result"]
    assert "not run" in rows["T5-31"]["result"]


def test_moving_the_person_moves_their_left():
    from curio_robot.demo.session import Scene, decide, seat_angle, seat_person

    with SimWorld(*PANE) as w:
        sc = SCENARIOS["perspective"](w, np.random.default_rng(0))
        scene = Scene("perspective", 0, sc.request, sc.expected_xy, sc.human_eye, sc.human_look)
        across = decide(w, scene, sc.request, True).result
        beside = seat_person(w, scene, 180.0)
        assert abs(abs(seat_angle(beside)) - 180.0) < 1e-6
        on = decide(w, beside, sc.request, True).result
        off = decide(w, beside, sc.request, False).result
    assert across["verdict"] == "correct" and on["verdict"] == "correct"
    assert on["target_id"] != across["target_id"]  # same words, other seat, other block
    assert off["verdict"] == "correct"  # beside the robot, their left is its left


def test_voice_phrases_are_what_the_resolver_reads():
    from curio_robot.demo import voice
    from curio_robot.resolver import parse_request

    for p in voice.requests():
        req = parse_request(p)
        assert req.color is not None  # every spoken request names a colour
    assert voice.to_command("curio compare") == "c"
    assert voice.to_command("curio walk around") == "seat 180"
    assert voice.to_command("curio hand me the red one on my left") == \
        "hand me the red one on my left"
    assert all(p.startswith("curio ") for p in voice.phrases())


@pytest.mark.skipif("not __import__('curio_robot.demo.voice').demo.voice.available()")
def test_voice_round_trip_on_windows(tmp_path):
    from curio_robot.demo import voice

    wav = tmp_path / "said.wav"
    voice.synthesize("Curio, give me the blue block closest to me", wav)
    got = []
    listener = voice.Listener(lambda c, t, conf: got.append(c), wave_file=wav)
    listener.start()
    listener.wait(60)
    time.sleep(0.3)
    listener.close()
    assert got == ["give me the blue block closest to me"]


@pytest.mark.parametrize("seat", [None, 90.0, 180.0, -150.0])
def test_handover_puts_the_chosen_block_in_front_of_the_person(seat):
    from curio_robot.demo.handover import HandOver
    from curio_robot.demo.session import Scene, decide, seat_person
    from curio_robot.sim.world import HOME_QPOS

    with SimWorld(*PANE, person_body=True) as w:
        sc = SCENARIOS["perspective"](w, np.random.default_rng(0))
        scene = Scene("perspective", 0, sc.request, sc.expected_xy, sc.human_eye, sc.human_look)
        if seat is not None:
            scene = seat_person(w, scene, seat)
        d = decide(w, scene, sc.request, True)
        h = HandOver(w, d.chosen, scene.eye, scene.look)
        start = w.model.body(h.body).pos.copy()
        events = []
        for _ in range(30 * 20):
            e = h.step(1 / 30)
            if e:
                events.append(e)
            if h.done:
                break
        end = w.model.body(h.body).pos.copy()
        assert h.result == "handed over" and events == ["attach", "release"]
        assert np.linalg.norm(start[:2] - d.chosen[:2]) < 0.02  # it took the chosen block
        assert np.linalg.norm(end[:2] - h.offer) < 0.02 and abs(end[2] - 0.0125) < 1e-6
        assert np.allclose(w.data.qpos[:6], HOME_QPOS)  # and went home
