import argparse
import sys
from pathlib import Path

from hailo_platform import VDevice
from hailo_platform.genai import LLM

repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break

if repo_root is not None:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import LLM_CHAT_APP, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH
from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)


def build_prompt(history, user_text):
    prompt = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant. Answer clearly and briefly."}]
        }
    ]

    for role, text in history:
        prompt.append({
            "role": role,
            "content": [{"type": "text", "text": text}]
        })

    prompt.append({
        "role": "user",
        "content": [{"type": "text", "text": user_text}]
    })

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Interactive LLM Chat Example")
    parser.add_argument("--hef-path", type=str, default=None, help="Path to HEF model file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum generated tokens")

    handle_list_models_flag(parser, LLM_CHAT_APP)
    args = parser.parse_args()

    hef_path = resolve_hef_path(args.hef_path, app_name=LLM_CHAT_APP, arch=HAILO10H_ARCH)
    if hef_path is None:
        logger.error("Failed to resolve HEF path for LLM model.")
        sys.exit(1)

    logger.info(f"Using HEF: {hef_path}")
    print(f"✓ Model file found: {hef_path}")

    vdevice = None
    llm = None
    history = []

    try:
        print("\n[1/2] Initializing Hailo device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("✓ Hailo device initialized")

        print("[2/2] Loading LLM model...")
        llm = LLM(vdevice, str(hef_path))
        print("✓ Model loaded successfully")

        print("\n" + "=" * 60)
        print("Interactive LLM Chat")
        print("Ketik pertanyaan lalu Enter")
        print("Command:")
        print("  /quit  -> keluar")
        print("  /clear -> hapus context")
        print("=" * 60)

        while True:
            try:
                user_text = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nKeluar.")
                break

            if not user_text:
                continue

            if user_text.lower() in {"/quit", "quit", "/exit", "exit"}:
                print("Keluar.")
                break

            if user_text.lower() == "/clear":
                history = []
                try:
                    llm.clear_context()
                except Exception:
                    pass
                print("Context cleared.")
                continue

            prompt = build_prompt(history, user_text)

            print("\nAssistant> ", end="", flush=True)

            try:
                response = llm.generate_all(
                    prompt=prompt,
                    temperature=args.temperature,
                    seed=42,
                    max_generated_tokens=args.max_tokens
                )

                if isinstance(response, str):
                    clean_response = response.split(". [{'type'")[0].strip()
                else:
                    clean_response = str(response).strip()

                print(clean_response)

                history.append(("user", user_text))
                history.append(("assistant", clean_response))

            except KeyboardInterrupt:
                print("\nGenerasi dihentikan.")
                continue
            except Exception as e:
                logger.error(f"Error during generation: {e}", exc_info=True)
                print(f"\n[ERROR] {e}")

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if llm:
            try:
                llm.clear_context()
                llm.release()
            except Exception as e:
                logger.warning(f"Error releasing LLM: {e}")

        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


if __name__ == "__main__":
    main()