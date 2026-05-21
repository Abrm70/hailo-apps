from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from audio_ring_buffer import AudioRingBuffer

if TYPE_CHECKING:
    import serial
    from vosk import Model


@dataclass
class AssistantRuntime:
    esp: Optional["serial.Serial"] = None
    wake_model: Optional["Model"] = None
    ring_buffer: AudioRingBuffer = field(default_factory=AudioRingBuffer)
