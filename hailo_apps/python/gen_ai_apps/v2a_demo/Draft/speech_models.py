from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SpeechModels:
    hailo_encoder: Optional[Any] = None
    hf_processor: Optional[Any] = None
    hf_decoder: Optional[Any] = None
    faster_whisper: Optional[Any] = None
