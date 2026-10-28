"""The live runtime: capture, perception, voice, the VLM and the arm as
concurrent stages, so nothing slow sits in front of the arm.

  main thread   capture @ capture_hz ──► Latest[frames]
  perception    detector + fusion + tracker ──► Latest[ObjectMap]   (+ hand interrupts)
  voice-in      VAD + streaming STT ──► Heard events
  orchestrator  events ──► Motion.go/stop, VoiceOut.say, VLM jobs
  vlm pool      VlmResolver on a map snapshot ──► VlmReply event
  motion        small IK steps toward the current goal ──► ArmEvent
  voice-out     TTS queue with barge-in

Capture stays on the main thread because MuJoCo's GL context belongs to the
thread that created it. The World itself makes capture safe against the arm."""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass

from curio_robot.episode_log import EpisodeLog
from curio_robot.events import Event
from curio_robot.perception import Detector
from curio_robot.runtime.motion import Motion
from curio_robot.runtime.orchestrator import Orchestrator
from curio_robot.runtime.perception_stage import PerceptionStage
from curio_robot.runtime.slot import Latest
from curio_robot.runtime.trace import Tracer
from curio_robot.types import CameraFrame, ObjectMap
from curio_robot.vlm import VlmResolver
from curio_robot.voice import VoiceIn, VoiceOut
from curio_robot.voice.scripted import PrintedVoice
from curio_robot.world import World


@dataclass(slots=True)
class RuntimeConfig:
    capture_hz: float = 15.0
    use_human_view: bool = True  # the Curio on/off switch
    arm_speed: float = 0.25  # m/s


class Runtime:
    def __init__(
        self,
        world: World,
        detector: Detector,
        *,
        voice_in: VoiceIn | None = None,
        voice_out: VoiceOut | None = None,
        vlm: VlmResolver | None = None,
        log: EpisodeLog | None = None,
        hand_in_workspace: Callable[[CameraFrame], bool] | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.world = world
        self.bus: queue.Queue[Event] = queue.Queue()
        self.frames: Latest[dict[str, CameraFrame]] = Latest()
        self.maps: Latest[ObjectMap] = Latest()
        self.tracer = Tracer()
        self.voice_in = voice_in
        self.voice_out = voice_out or PrintedVoice()
        self.motion = Motion(world, self.bus.put, speed=self.config.arm_speed)
        self.perception = PerceptionStage(
            self.frames, self.maps, detector, self.tracer,
            use_human_view=lambda: self.config.use_human_view,
            hand_in_workspace=hand_in_workspace, emit=self.bus.put,
        )
        self.brain = Orchestrator(
            self.bus, self.maps, self.motion, self.voice_out, self.tracer, vlm=vlm, log=log
        )

    def start(self) -> None:
        self.motion.start()
        self.perception.start()
        self.brain.start()
        if self.voice_in:
            self.voice_in.start(self.bus.put)

    def tick(self) -> None:
        """Capture once. Call from the thread that created the world."""
        t0 = time.perf_counter()
        self.frames.put(self.world.capture())
        self.tracer.frame("capture", (time.perf_counter() - t0) * 1000)

    def run(self, seconds: float, until: Callable[[], bool] | None = None) -> None:
        period = 1.0 / self.config.capture_hz
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and not (until and until()):
            t0 = time.perf_counter()
            self.tick()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def close(self) -> None:
        if self.voice_in:
            self.voice_in.close()
        self.brain.close()
        self.perception.close()
        self.motion.close()
        self.voice_out.close()
