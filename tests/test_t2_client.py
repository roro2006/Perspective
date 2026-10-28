"""Acceptance tests for T2-05 (grounding/client.py), against a fake server."""

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from curio_robot.grounding.client import VllmGrounder, letter_probs
from curio_robot.grounding.prompt import GroundingRequest


def _response(top, content="A"):
    return {"choices": [{
        "message": {"role": "assistant", "content": content},
        "logprobs": {"content": [{"token": content, "logprob": 0.0, "top_logprobs": [
            {"token": t, "logprob": lp} for t, lp in top
        ]}]},
    }]}


def test_softmax_over_allowed_letters():
    resp = _response([("A", math.log(0.6)), ("B", math.log(0.2)), ("Z", math.log(0.2))])
    assert letter_probs(resp, ["A", "B"]) == pytest.approx({"A": 0.75, "B": 0.25})


def test_tokens_are_normalized_and_merged():
    resp = _response([(" A", math.log(0.3)), ("a", math.log(0.3)), ("B ", math.log(0.4))])
    assert letter_probs(resp, ["A", "B"]) == pytest.approx({"A": 0.6, "B": 0.4})


def test_fallback_to_the_emitted_letter():
    resp = _response([("x", 0.0)], content=" b ")
    assert letter_probs(resp, ["A", "B"]) == {"B": 1.0}
    no_logprobs = {"choices": [{"message": {"content": "A"}}]}
    assert letter_probs(no_logprobs, ["A", "B"]) == {"A": 1.0}
    assert letter_probs(_response([("x", 0.0)], content="maybe"), ["A"]) == {}


class _Fake(BaseHTTPRequestHandler):
    seen: ClassVar[list] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append((self.path, body))
        out = json.dumps(_response([("B", math.log(0.9)), ("A", math.log(0.1))], "B")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Fake.seen = []
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


REQ = GroundingRequest(messages=[{"role": "user", "content": "x"}], letters={"B": "red_1", "A": "blue_1"})


def test_score_maps_letters_to_objects(server):
    answer = VllmGrounder(server, "qwen-vl").score(REQ)
    assert answer.probs == pytest.approx({"red_1": 0.9, "blue_1": 0.1})
    assert (answer.letter, answer.source, answer.error) == ("B", "local", None)
    assert answer.latency_ms > 0
    path, body = _Fake.seen[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "qwen-vl" and body["messages"] == REQ.messages
    assert body["max_tokens"] == 1 and body["temperature"] == 0
    assert body["logprobs"] is True and body["top_logprobs"] == 10
    assert body["guided_choice"] == ["A", "B"]


def test_choice_param_is_configurable(server):
    VllmGrounder(server, "m", choice_param="structured_choice").score(REQ)
    assert _Fake.seen[0][1]["structured_choice"] == ["A", "B"]
    assert "guided_choice" not in _Fake.seen[0][1]


def test_unreachable_never_raises():
    answer = VllmGrounder("http://127.0.0.1:9/v1", timeout_s=0.5).score(REQ)
    assert answer.probs == {} and answer.source == "unavailable" and answer.error
