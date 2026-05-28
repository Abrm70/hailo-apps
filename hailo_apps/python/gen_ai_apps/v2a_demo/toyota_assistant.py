"""
Toyota Hybrid Assistant for Hailo V2A Demo

Goal:
- Keep default V2A tools working.
- Add offline general Q&A fallback using Qwen.
- Use custom Toyota wake word from resources/toyota_wakeword.onnx.
- Speak every response using Piper TTS.

Run from:
    /home/cc/hailo-apps/hailo-apps/hailo_apps/python/gen_ai_apps/v2a_demo

Example:
    python toyota_assistant.py --wake-word-model resources/toyota_wakeword.onnx

Optional:
    python toyota_assistant.py --wake-word-model resources/toyota_wakeword.onnx --language en
    python toyota_assistant.py --wake-word-model resources/toyota_wakeword.onnx --language id
    python toyota_assistant.py --wake-word-model resources/toyota_wakeword.onnx --language ja
"""

import argparse
import logging
import re
import sys
import time
import subprocess
import os
import tempfile
import threading
import queue
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import sounddevice as sd
import soundfile as sf
from hailo_platform import VDevice
from hailo_platform.genai import LLM

# Local V2A modules
from listener import (
    WakeWordListener,
    SAMPLE_RATE,
    WAKE_WORD_THRESHOLD,
    WAKE_WORD_CONSECUTIVE,
    WAKE_SMOOTHING_FRAMES,
    WAKE_WARMUP_FRAMES,
)
from stt import STTEngine
from tool_selector import ToolSelector
from llm import LLMEngine
from tts import TTSEngine
from tools import run_tool

# Make repository imports work even when this file is run directly.
repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break
if repo_root is not None and str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.core.common.defines import (
    HAILO10H_ARCH,
    SHARED_VDEVICE_GROUP_ID,
    V2A_DEMO_APP,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import streaming

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"
DEFAULT_WAKE_WORD_MODEL = RESOURCES_DIR / "toyota_wakeword.onnx"

LOGGER_NAME = "toyota_assistant"
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = {"en", "id", "ja"}
NO_TOOL = "none"


def configure_logger(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
    )

    logger.handlers.clear()
    logger.addHandler(handler)

    # Make V2A internal modules use the same level/format.
    v2a_logger = logging.getLogger("v2a_demo")
    v2a_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    v2a_logger.propagate = False
    v2a_logger.handlers.clear()
    v2a_logger.addHandler(handler)

    return logger


def remove_wake_word_from_text(text: str) -> str:
    """Remove accidental wake word transcription from the command text."""
    cleaned = text.strip()
    patterns = [
        r"\bhalo\s+toyota\b",
        r"\bhello\s+toyota\b",
        r"\bhey\s+toyota\b",
        r"\bok\s+toyota\b",
        r"\btoyota\b",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?")
    return cleaned


def looks_like_only_wake_word(text: str) -> bool:
    cleaned = remove_wake_word_from_text(text)
    return len(cleaned.split()) == 0


def play_wake_beep(sample_rate: int = SAMPLE_RATE) -> None:
    """Play a short confirmation tone after the wake word is detected."""
    try:
        duration_s = 0.12
        frequency_hz = 880
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        audio = 0.25 * np.sin(2 * np.pi * frequency_hz * t)
        sd.play(audio.astype(np.float32), sample_rate)
        sd.wait()
    except Exception:
        # Beep is only a user cue. Do not stop the assistant if speaker playback fails.
        pass


class ToyotaWakeWordListener(WakeWordListener):
    """WakeWordListener with a short beep exactly when the wake word is detected."""

    def __init__(self, wake_word_model: str, audio_device: Optional[int] = None,
                 beep_enabled: bool = True):
        super().__init__(wake_word_model=wake_word_model, audio_device=audio_device)
        self._beep_enabled = beep_enabled

    def _wait_for_wake(self, frames) -> Optional[List[np.ndarray]]:
        """Listen for wake word. Returns pre-roll chunks on detection, None on timeout."""
        self._wake_word_model.reset()
        pre_roll = self._wake_warmup(frames)

        consecutive = 0
        score_history: List[float] = []
        wake_start = time.monotonic()
        v2a_logger = logging.getLogger("v2a_demo")

        for chunk in frames:
            pre_roll.append(chunk)
            scores = self._wake_word_model.predict(self._to_int16(chunk))

            score = max(scores.values())
            score_history.append(score)
            if len(score_history) > WAKE_SMOOTHING_FRAMES:
                score_history.pop(0)
            smoothed = float(np.mean(score_history))

            if smoothed >= WAKE_WORD_THRESHOLD:
                consecutive += 1
                if consecutive >= WAKE_WORD_CONSECUTIVE:
                    v2a_logger.info(f"Wake word detected (score={smoothed:.3f})")
                    if self._beep_enabled:
                        play_wake_beep()
                    return list(pre_roll)
            else:
                consecutive = 0

            if self._wake_timeout_s and time.monotonic() - wake_start >= self._wake_timeout_s:
                v2a_logger.info("Wake word timeout reached")
                return None

        return None



GENERAL_QUESTION_PATTERNS = [
    r"^\s*(what|who|why|how|when|where|can you tell me|tell me about|explain|describe|define)\b",
    r"^\s*(apa|siapa|kenapa|mengapa|bagaimana|kapan|dimana|jelaskan|ceritakan)\b",
    r"^\s*(何|誰|なぜ|どう|いつ|どこ|説明|教えて)\b",
]

TOOL_KEYWORDS = {
    "control_led": [
        "led", "lampu", "nyalakan led", "matikan led", "turn on the led",
        "turn off the led", "blink", "kedip", "berkedip", "ライト", "点滅",
    ],
    "get_weather": [
        "weather", "forecast", "temperature", "cuaca", "prakiraan", "hujan",
        "panas", "suhu udara", "天気", "気温", "雨",
    ],
    "system_check": [
        "system check", "check system", "cek sistem", "periksa sistem",
        "cpu", "ram", "memory", "disk", "storage", "suhu raspberry",
        "システム", "メモリ",
    ],
    "get_travel_time": [
        "travel time", "waktu tempuh", "drive from", "go from", "route from",
        "directions from", "how long from", "berapa lama dari", "jalan dari",
        "到着", "移動時間",
    ],
    "data_storage": [
        "remember", "save this", "store this", "my name is", "ingat",
        "simpan", "catat", "覚えて", "保存",
    ],
    "explain_tools": [
        "what can you do", "available tools", "your tools", "bisa apa",
        "fitur", "tools", "kemampuan", "何ができる",
    ],
}


def text_has_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def is_general_question(text: str) -> bool:
    t = text.strip().lower()
    return any(re.search(pattern, t) for pattern in GENERAL_QUESTION_PATTERNS)


def detect_explicit_tool_intent(text: str) -> Optional[str]:
    """Only return a tool when the user text clearly asks for that tool."""
    t = text.lower()

    for tool_name, keywords in TOOL_KEYWORDS.items():
        if text_has_any(t, keywords):
            if tool_name == "get_travel_time":
                has_route_shape = (
                    " from " in f" {t} " or
                    " dari " in f" {t} " or
                    " to " in f" {t} " or
                    " ke " in f" {t} "
                )
                if not has_route_shape:
                    return None
            return tool_name

    return None


def keyword_tool_fallback(text: str) -> Optional[str]:
    """
    Deterministic fallback for clear tool commands only.
    General questions must fall back to Qwen, not random V2A tools.
    """
    return detect_explicit_tool_intent(text)



def clean_llm_response(text: str) -> str:
    text = (text or "").strip()
    for marker in ["<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"]:
        text = text.replace(marker, "")
    return text.strip()


def clean_text_for_speech(text: str) -> str:
    text = clean_llm_response(text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[{}\[\]<>|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class GeneralQAEngine:
    """Offline general Q&A fallback using Qwen on Hailo."""

    def __init__(self, vdevice: VDevice, logger: logging.Logger):
        self.logger = logger

        model_path = resolve_hef_path(
            hef_path="Qwen2.5-Coder-1.5B-Instruct",
            app_name=V2A_DEMO_APP,
            arch=HAILO10H_ARCH,
        )
        if model_path is None:
            raise RuntimeError("Failed to resolve HEF path for Qwen2.5-Coder-1.5B-Instruct")

        self.llm = LLM(vdevice, str(model_path))

    def _prompt(self, user_text: str) -> List[dict]:
        system_prompt = (
            "You are Toyota's offline in-car AI assistant. "
            "Answer only the user's latest question. Do not continue any previous tool, JSON, weather, or parameter extraction task. "
            "Never output JSON examples unless the user explicitly asks for JSON. "
            "Do not write code unless the user explicitly asks for code. "
            "Use the same language as the user. If the user asks in Indonesian, answer in Indonesian. "
            "If the user asks in English, answer in English. If the user asks in Japanese, answer in Japanese. "
            "Keep the answer concise because it will be spoken aloud."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text.strip()},
        ]

    def clear_context(self):
        try:
            self.llm.clear_context()
        except Exception:
            pass

    def run(self, user_text: str) -> str:
        self.clear_context()
        try:
            response = self.llm.generate_all(
                prompt=self._prompt(user_text),
                max_generated_tokens=120,
                do_sample=False,
            )
        except TypeError:
            response = self.llm.generate_all(prompt=self._prompt(user_text))
        finally:
            self.clear_context()
        return (response or "").strip()

    def run_stream(self, user_text: str, token_callback: Optional[Callable[[str], None]] = None) -> str:
        self.clear_context()
        try:
            response = streaming.generate_and_stream_response(
                llm=self.llm,
                prompt=self._prompt(user_text),
                temperature=0.1,
                seed=42,
                max_tokens=120,
                prefix="Assistant: ",
                token_callback=token_callback,
                show_raw_stream=False,
            )
        finally:
            self.clear_context()
        return (response or "").strip()

    def close(self):
        try:
            self.llm.release()
        except Exception:
            pass

class ToyotaHybridAssistant:
    """
    Main hybrid pipeline:

    Wake word listener is handled in main().
    This class handles:
    STT -> ToolSelector -> Tool execution OR General Q&A -> TTS
    """

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        tts_output_path: Optional[str] = None,
        no_tts: bool = False,
        debug: bool = False,
    ):
        self.logger = configure_logger(debug)
        self.language = language
        self.tts_output_path = tts_output_path
        self.no_tts = no_tts
        self.debug = debug

        self.logger.info("Initializing Toyota hybrid assistant components...")

        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        self.vdevice = VDevice(params)

        self.stt = STTEngine(self.vdevice)
        self.tool_selector = ToolSelector(self.vdevice)
        self.tool_llm = LLMEngine(self.vdevice)
        self.general_qa = GeneralQAEngine(self.vdevice, self.logger)
        self.tts = None if no_tts else TTSEngine()

        self.logger.info("Toyota hybrid assistant initialized.")

    def __enter__(self):
        self.vdevice.__enter__()
        self.stt.__enter__()
        self.tool_selector.__enter__()
        self.tool_llm.__enter__()
        if self.tts:
            self.tts.__enter__()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for component in (self.stt, self.tool_selector, self.tool_llm, self.general_qa, self.tts):
            if component:
                try:
                    component.close()
                except Exception:
                    pass

        try:
            self.vdevice.release()
        except Exception:
            pass

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe command audio.

        Note:
        Hailo STTEngine default is English. For now:
        - --language en/id/ja passes that language code directly.
        - --language auto currently uses English as safe fallback because some
          Hailo Speech2Text builds do not support true language auto-detection.
        """
        language = DEFAULT_LANGUAGE if self.language == "auto" else self.language
        return self.stt.run(audio_data, language=language).strip()

    def select_tool(self, text: str) -> str:
        explicit_tool = detect_explicit_tool_intent(text)

        # General knowledge questions should go straight to offline Q&A
        # unless they clearly contain a supported tool keyword.
        if is_general_question(text) and explicit_tool is None:
            self.logger.info("General question detected. Skipping V2A tool selector.")
            return NO_TOOL

        # Very short noisy STT fragments often trigger random tools.
        if len(text.split()) <= 2 and explicit_tool is None:
            self.logger.info("Short/noisy command detected. Skipping V2A tool selector.")
            return NO_TOOL

        # For clear commands, prefer deterministic tool routing.
        if explicit_tool is not None:
            self.logger.info(f"Explicit tool intent detected: {explicit_tool}")
            return explicit_tool

        # If not a clear tool command, do not let semantic similarity force a tool.
        # This prevents examples like "What is Mount Fuji?" -> get_travel_time.
        self.logger.info("No explicit tool intent. Using offline general Q&A.")
        return NO_TOOL

    def _clear_all_llm_contexts(self):
        for obj in (self.tool_llm, self.general_qa):
            try:
                obj.llm.clear_context()
            except Exception:
                pass
            try:
                obj.clear_context()
            except Exception:
                pass

    def answer_with_tool(self, text: str, tool_name: str) -> str:
        params = self.tool_llm.run(text, tool_name)
        self.logger.debug(f"Extracted tool params: {params}")

        # Avoid invalid weather calls such as {"today": "true"}. Weather requires a city.
        if tool_name == "get_weather" and not params.get("location"):
            try:
                self.tool_llm.llm.clear_context()
            except Exception:
                pass
            return "Which city do you want the weather for?"

        try:
            response = run_tool(tool_name, params)
        finally:
            # Important: V2A parameter extraction uses cached prompt contexts.
            # Clear its active context so it cannot leak into the next general Q&A turn.
            try:
                self.tool_llm.llm.clear_context()
            except Exception:
                pass
        return response

    def answer_general_question(self, text: str) -> str:
        self.logger.info("No matching V2A tool. Using offline general Q&A.")
        self._clear_all_llm_contexts()
        return self.general_qa.run(text)

    def answer_general_question_streaming(self, text: str) -> str:
        self.logger.info("No matching V2A tool. Using offline general Q&A with streaming.")
        self._clear_all_llm_contexts()

        if self.no_tts or not self.tts:
            return self.general_qa.run_stream(text)

        sentence_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        response_parts: List[str] = []
        buffer = {"text": ""}

        def speaker_worker():
            while True:
                sentence = sentence_queue.get()
                if sentence is None:
                    sentence_queue.task_done()
                    break
                try:
                    self._play_tts_sentence(sentence)
                except Exception as e:
                    self.logger.warning(f"Streaming TTS sentence failed: {e}")
                finally:
                    sentence_queue.task_done()

        worker = threading.Thread(target=speaker_worker, daemon=True)
        worker.start()

        def flush_buffer(force: bool = False):
            candidate = buffer["text"].strip()
            if not candidate:
                return
            has_sentence_end = bool(re.search(r"[.!?。！？]\s*$", candidate))
            long_enough = len(candidate) >= 140
            if force or (has_sentence_end and len(candidate) >= 25) or long_enough:
                # Keep spoken chunks natural and avoid reading markdown fences.
                spoken = clean_text_for_speech(candidate)
                if spoken:
                    sentence_queue.put(spoken)
                buffer["text"] = ""

        def on_token(chunk: str):
            response_parts.append(chunk)
            buffer["text"] += chunk
            flush_buffer(force=False)

        raw = self.general_qa.run_stream(text, token_callback=on_token)
        flush_buffer(force=True)
        sentence_queue.put(None)
        sentence_queue.join()

        collected = "".join(response_parts).strip()
        return collected or raw

    def _play_wav_file(self, wav_path: Path):
        try:
            subprocess.run(["paplay", str(wav_path)], check=False)
        except FileNotFoundError:
            subprocess.run(["aplay", str(wav_path)], check=False)

    def _play_tts_sentence(self, text: str):
        text = clean_text_for_speech(text)
        if not text:
            return
        tts_result = self.tts.run(text.strip())
        if not tts_result:
            self.logger.warning("TTS produced no audio.")
            return
        audio_array, sample_rate = tts_result
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = Path(tmp.name)
        tmp.close()
        try:
            sf.write(str(wav_path), audio_array, sample_rate)
            self._play_wav_file(wav_path)
        finally:
            try:
                wav_path.unlink()
            except Exception:
                pass

    def speak(self, text: str):
        if self.no_tts or not self.tts or not text.strip():
            return

        self.logger.info("Generating speech...")
        text = clean_text_for_speech(text)
        if not text:
            return

        tts_result = self.tts.run(text.strip())
        if not tts_result:
            self.logger.warning("TTS produced no audio.")
            return

        audio_array, sample_rate = tts_result
        output_path = self.tts_output_path
        delete_after_play = False

        if output_path:
            wav_path = Path(output_path)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = Path(tmp.name)
            tmp.close()
            delete_after_play = True

        sf.write(str(wav_path), audio_array, sample_rate)

        if self.tts_output_path:
            self.logger.info(f"TTS saved to: {wav_path}")
            return

        try:
            self._play_wav_file(wav_path)
        finally:
            if delete_after_play:
                try:
                    wav_path.unlink()
                except Exception:
                    pass

    def process_audio(self, audio_data: np.ndarray) -> str:
        timings = {}

        t0 = time.perf_counter()
        user_text = self.transcribe(audio_data)
        timings["stt"] = time.perf_counter() - t0

        if not user_text:
            self.logger.info("No speech detected.")
            return ""

        self.logger.info(f"Transcription: {user_text}")

        if looks_like_only_wake_word(user_text):
            self.logger.info("Only wake word detected, ignoring.")
            return ""

        clean_text = remove_wake_word_from_text(user_text)
        if clean_text != user_text:
            self.logger.info(f"Command after wake-word cleanup: {clean_text}")

        t0 = time.perf_counter()
        tool_name = self.select_tool(clean_text)
        timings["tool_select"] = time.perf_counter() - t0
        self.logger.info(f"Selected tool: {tool_name}")

        if tool_name != NO_TOOL:
            t0 = time.perf_counter()
            response = self.answer_with_tool(clean_text, tool_name)
            timings["tool_or_qa"] = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            response = self.answer_general_question_streaming(clean_text)
            timings["tool_or_qa"] = time.perf_counter() - t0

        response = clean_llm_response(response or "")
        self.logger.info(f"Assistant response: {response}")

        t0 = time.perf_counter()
        if tool_name != NO_TOOL:
            self.speak(response)
        # For general Q&A, TTS has already been streamed sentence-by-sentence.
        timings["tts"] = time.perf_counter() - t0

        self._log_timing(timings)
        return response

    def _log_timing(self, timings: dict):
        self.logger.info("---- Toyota Assistant Performance ----")
        for key, value in timings.items():
            self.logger.info(f"{key:<14}: {value:>6.3f} s")
        self.logger.info(f"{'total':<14}: {sum(timings.values()):>6.3f} s")
        self.logger.info("---------------------------------------")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toyota Hybrid Assistant: V2A tools + offline Q&A")

    parser.add_argument(
        "--wake-word-model",
        default=str(DEFAULT_WAKE_WORD_MODEL),
        help="Path to custom ONNX wake word model. Default: resources/toyota_wakeword.onnx",
    )

    parser.add_argument(
        "--audio-input-path",
        default=None,
        help="Optional WAV file input for one-shot testing.",
    )

    parser.add_argument(
        "--audio-output-path",
        default=None,
        help="Optional path to save TTS output instead of playing it.",
    )

    parser.add_argument(
        "--audio-device",
        type=int,
        default=None,
        help="PortAudio microphone device index. Use: python -m sounddevice",
    )

    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="Show available audio devices and exit.",
    )

    parser.add_argument(
        "--no-wake-beep",
        action="store_true",
        help="Disable the short beep after wake word is detected.",
    )

    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        choices=["auto", "en", "id", "ja"],
        help="STT language. Use en/id/ja. 'auto' currently falls back to en if unsupported.",
    )

    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable speaker output.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs.",
    )

    return parser


def main():
    args = create_parser().parse_args()
    logger = configure_logger(args.debug)

    wake_model_path = Path(args.wake_word_model)
    if not wake_model_path.exists():
        raise FileNotFoundError(f"Wake word model not found: {wake_model_path}")

    if args.list_audio_devices:
        print_audio_devices(logger)
        return

    listener = ToyotaWakeWordListener(
        wake_word_model=str(wake_model_path),
        audio_device=args.audio_device,
        beep_enabled=not args.no_wake_beep,
    )

    # Fail fast if the microphone is not configured. This prevents endless repeated logs.
    try:
        listener._validate_audio_device()
    except Exception as e:
        logger.error(str(e))
        print_audio_devices(logger)
        logger.error("Choose the correct microphone index, then run again with: --audio-device <index>")
        return

    logger.info("Starting Toyota Hybrid Assistant.")
    logger.info(f"Wake word model: {wake_model_path}")
    logger.info(f"STT language: {args.language}")

    with ToyotaHybridAssistant(
        language=args.language,
        tts_output_path=args.audio_output_path,
        no_tts=args.no_tts,
        debug=args.debug,
    ) as assistant:
        if args.audio_input_path:
            logger.info("Processing one audio file...")
            audio = listener.listen_from_file(args.audio_input_path)
            if len(audio) > 0:
                assistant.process_audio(audio)
            return

        logger.info("Continuous wake-word mode started. Press Ctrl+C to exit.")
        logger.info("Standby: say 'Halo Toyota'.")
        try:
            while True:
                audio = listener.listen()

                if len(audio) == 0:
                    logger.warning("No command audio captured. Returning to standby.")
                    time.sleep(0.5)
                    continue

                try:
                    assistant.process_audio(audio)
                except Exception as e:
                    logger.exception(f"Assistant processing error: {e}")

                logger.info("Standby: say 'Halo Toyota'.")
        except KeyboardInterrupt:
            logger.info("Shutdown requested.")


if __name__ == "__main__":
    main()

