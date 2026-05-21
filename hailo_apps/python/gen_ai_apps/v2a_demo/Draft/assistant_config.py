from dataclasses import dataclass
from typing import Optional


@dataclass
class AssistantConfig:
    use_hailo_encoder: bool = True
    hailo_encoder_hef: str = "base-whisper-encoder-5s_h8l.hef"
    hailo_window_seconds: int = 5
    use_google_stt: bool = False
    decoder_local_dir: Optional[str] = None
    send_to_webserver: bool = True
    web_server_url: str = "http://10.0.0.76:5000/task"
    use_esp: bool = True
