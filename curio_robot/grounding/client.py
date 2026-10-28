"""Ask the served VLM for one letter and read the probabilities of every
candidate letter from the first token's logprobs. Task T2-05."""

from __future__ import annotations

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
    raise NotImplementedError("T2-05")


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
        raise NotImplementedError("T2-05")

    def score(self, request: GroundingRequest) -> GroundingAnswer:
        """letter_probs mapped through request.letters to object ids."""
        raise NotImplementedError("T2-05")
