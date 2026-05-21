import argparse
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from hailo_platform import VDevice
from hailo_platform.genai import LLM

repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break
if repo_root is not None:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.core.common.defines import (
    CHUNK_SIZE,
    HAILO10H_ARCH,
    LLM_PROMPT_PREFIX,
    SHARED_VDEVICE_GROUP_ID,
    TARGET_SR,
    VOICE_ASSISTANT_APP,
    VOICE_ASSISTANT_MODEL_NAME,
)
from hailo_apps.python.core.common.hailo_logger import add_logging_cli_args, init_logging, level_from_args
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import streaming
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

DEFAULT_WAKE_PHRASES = ("hello toyota", "ok toyota")
DEFAULT_TIRE_VIDEO = "/home/cc/videos/gantiban.mp4"
TIRE_PATTERNS = (
    "flat tire",
    "my tire is flat",
    "how to change a tire",
    "how to change tire",
    "change tire",
    "replace tire",
    "tire change",
    "tire is flat",
    "punctured tire",
)
COMMON_WAKE_FALSE_POSITIVES = {
    "the", "i", "a", "thank you", "thanks", "toy", "toyota toyota", "hello", "okay"
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    words = text.split()
    dedup_words = []
    for word in words:
        compact = _dedupe_compound_token(word)
        if not dedup_words or dedup_words[-1] != compact:
            dedup_words.append(compact)
    return " ".join(dedup_words)


def _dedupe_compound_token(token: str) -> str:
    for size in range(1, len(token) // 2 + 1):
        if len(token) % size == 0:
            piece = token[:size]
            if piece * (len(token) // size) == token and len(token) // size > 1:
                return piece
    return token


def is_valid_command_text(text: str) -> bool:
    if not text or text in COMMON_WAKE_FALSE_POSITIVES:
        return False
    if len(text) < 4:
        return False
    if len(text.split()) < 2:
        return False
    if not re.search(r"[a-z]", text):
        return False
    return True


def open_video(video_path: str, preferred_player: Optional[str] = None) -> bool:
    player_candidates = []
    if preferred_player:
        player_candidates.append(preferred_player)
    player_candidates.extend(["mpv", "vlc", "xdg-open"])

    for candidate in player_candidates:
        exe = shutil.which(candidate)
        if not exe:
            continue
        try:
            if Path(exe).name == "mpv":
                subprocess.Popen([exe, "--fs", video_path])
            elif Path(exe).name == "vlc":
                subprocess.Popen([exe, "--fullscreen", video_path])
            else:
                subprocess.Popen([exe, video_path])
            return True
        except Exception as exc:
            logger.warning("Failed to launch %s: %s", candidate, exc)
    return False


class NativeWakeVoiceAssistantV2:
    def __init__(self, args):
        self.args = args
        self.abort_event = threading.Event()
        self.lock = threading.Lock()
        self.running = True

        self.recorder = AudioRecorder(debug=args.debug)
        self.player = args.video_player
        self.video_path = args.tire_video
        self.wake_phrases = {normalize_text(x) for x in args.wake_phrase}

        self.state = "WAIT_WAKE"
        self.cooldown_until = 0.0

        self.vad = VoiceActivityDetector(
            sample_rate=TARGET_SR,
            chunk_size=CHUNK_SIZE,
            aggressiveness=args.vad_aggressiveness,
            min_speech_duration_ms=args.min_speech_ms,
            min_silence_duration_ms=args.min_silence_ms,
            energy_threshold=args.vad_energy_threshold,
            warmup_chunks=args.vad_warmup_chunks,
        )

        self.segment_audio: List[np.ndarray] = []
        self.segment_has_speech = False
        self.prev_vad_state = False
        self.command_started_at = 0.0

        print("Initializing AI components... (this may take a moment)")
        with redirect_stderr(StringIO()):
            params = VDevice.create_params()
            params.group_id = SHARED_VDEVICE_GROUP_ID
            self.vdevice = VDevice(params)
            self.s2t = SpeechToTextProcessor(self.vdevice)
            self.llm = None
            if not args.no_llm:
                model_path = resolve_hef_path(
                    hef_path=VOICE_ASSISTANT_MODEL_NAME,
                    app_name=VOICE_ASSISTANT_APP,
                    arch=HAILO10H_ARCH,
                )
                if model_path is None:
                    raise RuntimeError(
                        "Failed to resolve HEF path for the voice assistant LLM. "
                        "Run hailo-download-resources or use --no-llm for routing-only test."
                    )
                self.llm = LLM(self.vdevice, str(model_path))
        print("✅ Components ready")

    def run(self):
        self.recorder.start(stream_callback=self._audio_callback)
        print("\nHands-free mode active (speech-segment based)")
        print(f"Wake phrases: {', '.join(self.wake_phrases)}")
        print("Waiting for full speech segment before STT, so it should be much less spammy.")
        print("Press Ctrl+C to exit\n")

        try:
            while self.running:
                now = time.monotonic()
                if self.state == "LISTEN_COMMAND" and self.command_started_at:
                    if now - self.command_started_at >= self.args.max_command_sec:
                        self._force_finalize_command("max duration")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.close()

    def close(self):
        self.running = False
        try:
            self.recorder.close()
        except Exception:
            pass
        try:
            if self.llm is not None:
                self.llm.release()
        except Exception:
            pass

    def _audio_callback(self, chunk: np.ndarray):
        try:
            if time.monotonic() < self.cooldown_until:
                return

            chunk = self._prepare_chunk(chunk)
            if chunk.size == 0:
                return

            is_speech, energy = self.vad.process(chunk)
            if self.args.debug:
                logger.debug("state=%s energy=%.4f is_speech=%s", self.state, energy, is_speech)

            with self.lock:
                if self.state in ("WAIT_WAKE", "LISTEN_COMMAND"):
                    self.segment_audio.append(chunk)
                    if is_speech:
                        self.segment_has_speech = True

                if is_speech and not self.prev_vad_state:
                    if self.state == "WAIT_WAKE":
                        print("\n[WAKE-LISTEN] speech detected")
                    elif self.state == "LISTEN_COMMAND":
                        print("\n[COMMAND] speech detected")

                if not is_speech and self.prev_vad_state:
                    # Full speech segment ended -> evaluate once
                    audio = self._consume_segment_audio_locked()
                    prev_state = self.state
                    self.prev_vad_state = is_speech
                    self._handle_finished_segment(audio, prev_state)
                    return

                # trim silence-only backlog while waiting wake
                if self.state == "WAIT_WAKE" and not self.segment_has_speech and len(self.segment_audio) > self.args.max_idle_chunks:
                    self.segment_audio = self.segment_audio[-self.args.keep_idle_chunks:]

                self.prev_vad_state = is_speech
        except Exception as exc:
            logger.error("Audio callback failed: %s", exc)

    def _prepare_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim > 1:
            chunk = chunk.flatten()
        if self.recorder.device_sr != TARGET_SR:
            num_samples = int(len(chunk) * TARGET_SR / self.recorder.device_sr)
            chunk = np.interp(
                np.linspace(0.0, 1.0, num_samples, endpoint=False),
                np.linspace(0.0, 1.0, len(chunk), endpoint=False),
                chunk,
            ).astype(np.float32)
        return chunk.astype(np.float32, copy=False)

    def _consume_segment_audio_locked(self) -> np.ndarray:
        valid = [c for c in self.segment_audio if c is not None and len(c) > 0]
        self.segment_audio = []
        self.segment_has_speech = False
        if not valid:
            return np.array([], dtype=np.float32)
        return np.concatenate(valid, axis=0).astype(np.float32, copy=False)

    def _handle_finished_segment(self, audio: np.ndarray, prev_state: str):
        if audio.size == 0:
            return

        if prev_state == "WAIT_WAKE":
            self._evaluate_wake_segment(audio)
        elif prev_state == "LISTEN_COMMAND":
            self._finalize_command(audio, "silence")

    def _evaluate_wake_segment(self, audio: np.ndarray):
        duration_sec = len(audio) / TARGET_SR
        if duration_sec < self.args.min_wake_audio_sec:
            return

        text = normalize_text(self.s2t.transcribe(audio, language=self.args.stt_language))
        if not text:
            return
        logger.info("Wake segment: %s", text)
        print(f"[WAKE?] {text}")

        if text in COMMON_WAKE_FALSE_POSITIVES:
            return

        if text in self.wake_phrases:
            print(f"[WAKE] detected: {text}")
            self.state = "LISTEN_COMMAND"
            self.command_started_at = time.monotonic()
            self.vad.reset()
            self.prev_vad_state = False
            print("[LISTEN] Speak your command...")

    def _force_finalize_command(self, reason: str):
        with self.lock:
            audio = self._consume_segment_audio_locked()
        if audio.size:
            self._finalize_command(audio, reason)
        else:
            self._reset_to_wait()

    def _finalize_command(self, audio: np.ndarray, reason: str):
        self.state = "PROCESSING"
        print(f"[PROCESS] finishing command ({reason})")

        try:
            text = normalize_text(self.s2t.transcribe(audio, language=self.args.stt_language))
            print(f"You: {text or '<empty>'}")

            if not is_valid_command_text(text):
                print("[INFO] Ignored: command too short or noisy")
                self._reset_to_wait()
                return

            if self._route_local_intent(text):
                self._reset_to_wait()
                return

            if self.llm is None:
                print("[INFO] No local route matched and --no-llm active")
                self._reset_to_wait()
                return

            print("\nLLM response:\n")
            prompt_text = LLM_PROMPT_PREFIX + text
            formatted_prompt = [{"role": "user", "content": prompt_text}]
            self.abort_event.clear()
            streaming.generate_and_stream_response(
                llm=self.llm,
                prompt=formatted_prompt,
                prefix="",
                show_raw_stream=self.args.debug,
                abort_callback=self.abort_event.is_set,
            )
            print()
        except Exception as exc:
            logger.error("Command processing failed: %s", exc)
            print(f"[ERROR] {exc}")
        finally:
            self._reset_to_wait()

    def _route_local_intent(self, text: str) -> bool:
        if any(pattern in text for pattern in TIRE_PATTERNS):
            print(f"[ROUTE] flat-tire SOP -> {self.video_path}")
            if not Path(self.video_path).exists():
                print(f"[WARN] Video not found: {self.video_path}")
                return True
            opened = open_video(self.video_path, preferred_player=self.player)
            if not opened:
                print("[WARN] No video player found (tried mpv, vlc, xdg-open)")
            return True
        return False

    def _reset_to_wait(self):
        with self.lock:
            self.state = "WAIT_WAKE"
            self.segment_audio = []
            self.segment_has_speech = False
            self.prev_vad_state = False
            self.command_started_at = 0.0
            self.vad.reset()
        self.cooldown_until = time.monotonic() + self.args.post_command_cooldown_sec
        print("[READY] Waiting for wake phrase...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hands-free native hailo-apps voice assistant with wake phrase and local routing (segment-based)"
    )
    add_logging_cli_args(parser)
    parser.add_argument("--wake-phrase", action="append", default=list(DEFAULT_WAKE_PHRASES),
                        help="Wake phrase. Can be repeated. Default: hello toyota, ok toyota")
    parser.add_argument("--stt-language", default="en", help="Speech-to-text language. Default: en")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM and test routing only")
    parser.add_argument("--tire-video", default=DEFAULT_TIRE_VIDEO, help="Local SOP video path")
    parser.add_argument("--video-player", default=None, help="Preferred video player binary, for example mpv")
    parser.add_argument("--min-wake-audio-sec", type=float, default=0.6, help="Minimum speech segment duration for wake STT")
    parser.add_argument("--max-command-sec", type=float, default=8.0, help="Maximum command length")
    parser.add_argument("--post-command-cooldown-sec", type=float, default=1.0, help="Cooldown after each processed command")
    parser.add_argument("--min-speech-ms", type=int, default=220, help="VAD minimum speech duration in ms")
    parser.add_argument("--min-silence-ms", type=int, default=900, help="VAD minimum silence duration in ms")
    parser.add_argument("--vad-aggressiveness", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--vad-energy-threshold", type=float, default=0.03,
                        help="Raise this if it still reacts to noise. Try 0.03 to 0.05")
    parser.add_argument("--vad-warmup-chunks", type=int, default=12, help="Ignore initial chunks after start/reset")
    parser.add_argument("--max-idle-chunks", type=int, default=40, help="Trim idle waiting buffer")
    parser.add_argument("--keep-idle-chunks", type=int, default=8, help="Keep this many chunks while idle")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    init_logging(level=level_from_args(args))
    app = NativeWakeVoiceAssistantV2(args)
    app.run()


if __name__ == "__main__":
    main()

