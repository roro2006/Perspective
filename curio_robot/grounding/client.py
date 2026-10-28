"""Ask the served VLM for one letter and read the probabilities of every
candidate letter from the first token's logprobs. Task T2-05."""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass

from curio_robot.grounding.prompt import GroundingRequest

DEFAULT_URL = "http://127.0.0.1:8000/v1"  # vLLM in WSL2 on the 7900 XT


@dataclass(frozen=True, slots=True)
class GroundingAnswer:
    probs: dict[str, float]  # object_id -> probability, sums to 1 ({} on failure)
    letter: str | None  # the letter the model actually emitted
    latency_ms: float
    source: str  # endpoint name, or "unavailable"
    error: str | None = None


def letter_probs(response: dict, letters: list[str]) -> dict[str, float]:
    """P(letter) for the allowed letters, from an OpenAI chat completion with
    logprobs: response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    is a list of {"token": str, "logprob": float}.

    Normalize each token with .strip().upper(); keep allowed letters only (if
    a letter appears twice, sum their probabilities); softmax-normalize what's
    kept so it sums to 1. If no allowed letter is in top_logprobs (or logprobs
    are missing), fall back to the message content: if
    choices[0]["message"]["content"].strip().upper() is an allowed letter,
    return {letter: 1.0}; else {}."""
    allowed_set = set(letters)
    masses: dict[str, float] = {}

    try:
        top_logprobs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        for item in top_logprobs:
            token = item["token"].strip().upper()
            if token in allowed_set:
                prob = math.exp(item["logprob"])
                masses[token] = masses.get(token, 0.0) + prob
    except (KeyError, TypeError, IndexError):
        pass

    if masses:
        total = sum(masses.values())
        if total > 0:
            return {k: v / total for k, v in masses.items()}
        return {}

    try:
        content = response["choices"][0]["message"]["content"].strip().upper()
        if content in allowed_set:
            return {content: 1.0}
    except (KeyError, TypeError, IndexError):
        pass

    return {}


class VllmGrounder:
    """One OpenAI-compatible endpoint. The request body is
    {"model", "messages", "max_tokens": 1, "temperature": 0, "logprobs": true,
     "top_logprobs": 10, <choice_param>: sorted(request.letters)}
    where choice_param defaults to "guided_choice" (vLLM's extra parameter;
    newer builds may want a different name, confirmed in T2-10). POST to
    <base_url>/chat/completions with urllib, JSON, timeout_s. Never raises:
    any error -> GroundingAnswer({}, None, latency, "unavailable", str(error))."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = "qwen-vl",
        *,
        name: str = "local",
        timeout_s: float = 2.0,
        choice_param: str = "guided_choice",
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._name = name
        self._timeout_s = timeout_s
        self._choice_param = choice_param

    def score(self, request: GroundingRequest) -> GroundingAnswer:
        """letter_probs mapped through request.letters to object ids."""
        start = time.perf_counter()
        try:
            response = self._post(request)
            letters = sorted(request.letters)
            probs_by_letter = letter_probs(response, letters)
            probs = {request.letters[letter]: p for letter, p in probs_by_letter.items()}
            emitted = response["choices"][0]["message"]["content"].strip().upper()
            return GroundingAnswer(
                probs=probs,
                letter=emitted if emitted in request.letters else None,
                latency_ms=_elapsed_ms(start),
                source=self._name,
            )
        except Exception as e:  # noqa: BLE001 - the grounder never raises; the caller falls back
            return GroundingAnswer(
                probs={},
                letter=None,
                latency_ms=_elapsed_ms(start),
                source="unavailable",
                error=str(e) or type(e).__name__,
            )

    def _post(self, request: GroundingRequest) -> dict:
        body = {
            "model": self._model,
            "messages": request.messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 10,
            self._choice_param: sorted(request.letters),
        }
        req = urllib.request.Request(
            self._base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            return json.load(resp)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
