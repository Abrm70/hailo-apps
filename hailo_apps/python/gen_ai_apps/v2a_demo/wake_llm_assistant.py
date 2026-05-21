import argparse
import logging
import sys
import threading
import tempfile
import os
import subprocess
from pathlib import Path
from io import StringIO
from contextlib import redirect_stderr

import soundfile as sf

from hailo_platform import VDevice
from hailo_platform.genai import LLM

from listener import WakeWordListener
from tts import TTSEngine

repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break

if repo_root is not None:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.core.common.defines import (
    LLM_PROMPT_PREFIX,
    SHARED_VDEVICE_GROUP_ID,
    HAILO10H_ARCH,
    VOICE_ASSISTANT_APP,
    VOICE_ASSISTANT_MODEL_NAME,
)

from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
from hailo_apps.python.gen_ai_apps.gen_ai_utils.llm_utils import streaming


RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def configure_logger(debug=False):
    logger = logging.getLogger("wake_llm_assistant")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
    )

    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


class WakeLLMAssistant:
    def __init__(self, logger, no_tts=False, debug=False):
        self.logger = logger
        self.debug = debug
        self.no_tts = no_tts
        self.abort_event = threading.Event()

        self.logger.info("Initializing AI components...")

        with redirect_stderr(StringIO()):
            params = VDevice.create_params()
            params.group_id = SHARED_VDEVICE_GROUP_ID
            self.vdevice = VDevice(params)

            self.s2t = SpeechToTextProcessor(self.vdevice)

            model_path = resolve_hef_path(
                hef_path=VOICE_ASSISTANT_MODEL_NAME,
                app_name=VOICE_ASSISTANT_APP,
                arch=HAILO10H_ARCH,
            )

            if model_path is None:
                raise RuntimeError("LLM HEF model not found.")

            self.llm = LLM(self.vdevice, str(model_path))

            self.tts = None
            if not no_tts:
                self.tts = TTSEngine()

        self.logger.info("AI components ready.")

    def speak_text(self, text: str):
        if self.no_tts or not self.tts or not text.strip():
            return

        self.logger.info("Generating speech...")

        tts_result = self.tts.run(text.strip())
        if not tts_result:
            self.logger.warning("TTS returned no audio.")
            return

        audio_array, sample_rate = tts_result

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        sf.write(tmp_path, audio_array, sample_rate)

        try:
            subprocess.run(["paplay", tmp_path], check=True)
        except Exception as e:
            self.logger.error(f"Failed to play TTS audio: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def process_audio(self, audio):
        self.abort_event.clear()

        user_text = self.s2t.transcribe(audio)
        if not user_text:
            self.logger.info("No speech detected.")
            return

        self.logger.info(f"Transcription: {user_text}")
        clean_text = user_text.lower().strip()

        wake_noise = clean_text.replace(".", "").replace(",", "").strip()
        wake_words = ["hello toyota", "halo toyota", "hey toyota", "toyota"]

        if any(w in wake_noise for w in wake_words) and len(clean_text.split()) < 8:
            self.logger.info("Only wake word detected, ignoring command.")
            return

        prompt_text = LLM_PROMPT_PREFIX + user_text
        formatted_prompt = [{"role": "user", "content": prompt_text}]

        response_parts = []

        def collect_callback(chunk: str):
            response_parts.append(chunk)

        self.logger.info("Generating LLM response...")
        print("\nAssistant: ", end="", flush=True)

        streaming.generate_and_stream_response(
            llm=self.llm,
            prompt=formatted_prompt,
            prefix="",
            show_raw_stream=self.debug,
            token_callback=collect_callback,
            abort_callback=self.abort_event.is_set,
        )

        print()

        full_response = "".join(response_parts).strip()

        if full_response:
            self.speak_text(full_response)
        else:
            self.logger.warning("LLM produced empty response.")

    def close(self):
        try:
            if self.tts:
                self.tts.close()
        except Exception:
            pass

        try:
            self.llm.release()
        except Exception:
            pass

        try:
            self.vdevice.release()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Wakeword + offline LLM voice assistant")

    parser.add_argument(
        "--wake-word-model",
        default=str(RESOURCES_DIR / "hello_toyota.onnx"),
        help="Path to custom ONNX wake word model",
    )

    parser.add_argument(
        "--audio-device",
        type=int,
        default=None,
        help="Microphone device index",
    )

    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS output",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()
    logger = configure_logger(args.debug)

    listener = WakeWordListener(
        wake_word_model=args.wake_word_model,
        audio_device=args.audio_device,
    )

    assistant = WakeLLMAssistant(
        logger=logger,
        no_tts=args.no_tts,
        debug=args.debug,
    )

    logger.info("Starting wakeword LLM assistant.")
    logger.info(f"Listening for wake word model: {args.wake_word_model}")

    try:
        while True:
            logger.info("Listening for wake word...")
            audio = listener.listen()

            if len(audio) > 0:
                logger.info(f"Command audio captured: {len(audio)} samples")
                try:
                    assistant.process_audio(audio)
                except Exception as e:
                    logger.exception(f"Assistant error: {e}")

            logger.info("Ready for next wake word.")
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    finally:
        assistant.close()


if __name__ == "__main__":
    main()