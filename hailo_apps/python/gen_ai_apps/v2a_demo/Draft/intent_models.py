from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class IntentModels:
    tokenizer: "AutoTokenizer"
    model: "AutoModelForSequenceClassification"
    id_to_label: Dict[int, str]
