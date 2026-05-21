from dataclasses import dataclass


@dataclass
class ProgramState:
    last_command: str = "nyalakan_lampu"
    current_light_state: str = "ON"
    gender: str = "pria"
    predicted_language_from_wake_word: str = "Indonesian"
