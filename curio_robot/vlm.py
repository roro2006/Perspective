"""The slow half of the cascade: a VLM reads the object map as text once per
utterance and answers with [TARGET:id].

Any OpenAI-compatible chat endpoint works. Default: llama.cpp's llama-server
on the RX 7900 XT (Vulkan or HIP build) serving Qwen3-VL-8B; the lab GPU's
endpoint is the next fallback, the rule-based resolver the last one."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass

from curio_robot.resolver import parse_target_tag, vlm_prompt
from curio_robot.types import ObjectMap

LOCAL_URL = "http://127.0.0.1:8080/v1"


@dataclass(slots=True)
class ChatEndpoint:
    name: str
    base_url: str = LOCAL_URL
    model: str = "qwen3-vl-8b"
    api_key: str | None = None
    timeout_s: float = 2.0

    def complete(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 64,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]


@dataclass(frozen=True, slots=True)
class VlmAnswer:
    object_id: str | None
    spoken: str
    source: str  # endpoint name, or "unavailable" when every endpoint failed
    latency_ms: float
    errors: tuple[str, ...] = ()


class VlmResolver:
    """Tries each endpoint in order. Returns source="unavailable" rather than
    raising, so the caller keeps the rule-based target it already acted on."""

    def __init__(self, endpoints: list[ChatEndpoint]) -> None:
        self._endpoints = endpoints

    def resolve(self, text: str, object_map: ObjectMap) -> VlmAnswer:
        prompt = vlm_prompt(text, object_map)
        known = {o.object_id for o in object_map.objects}
        errors = []
        t0 = time.perf_counter()
        for ep in self._endpoints:
            try:
                spoken, object_id = parse_target_tag(ep.complete(prompt))
            except Exception as e:  # noqa: BLE001 - network, HTTP and JSON errors all mean "next"
                errors.append(f"{ep.name}: {e}")
                continue
            if object_id is not None and object_id not in known:
                errors.append(f"{ep.name}: unknown id {object_id}")
                object_id = None
            ms = (time.perf_counter() - t0) * 1000
            return VlmAnswer(object_id, spoken, ep.name, ms, tuple(errors))
        return VlmAnswer(None, "", "unavailable", (time.perf_counter() - t0) * 1000, tuple(errors))
