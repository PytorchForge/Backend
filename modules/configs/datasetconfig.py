from pydantic import BaseModel, PrivateAttr
from typing import Optional, Dict, Literal,Any

class DatasetConfig(BaseModel):
    split_type: Literal["include", "exclude"] = "include"
    label_type: Literal["folder-name", "file", "csv", "none"] = "folder-name"
    label_map: Optional[Dict[str, int]] = None
    return_format: Literal["raw", "dict", "tuple"] = "dict"

    # Task-specific
    tokenizer_type: Optional[str] = None  # e.g., AutoTokenizer for now only
    tokenizer_params: Optional[Dict[str, Any]] = None  # TEMPORARY: placeholder
    audio_duration: Optional[float] = None  # clip/pad (seconds)

    # Flags
    include_prompt: Optional[bool] = None
    include_target: Optional[bool] = None
    text_pair: Optional[bool] = None
    audio_pair: Optional[bool] = None
    multi_class: Optional[bool] = None
    multi_label: Optional[bool] = None
    binary: Optional[bool] = None

    # TEMPORARY 
    _note: str = PrivateAttr(default="Tokenization/audio-specific logic not yet fully defined")



class DataloaderConfig(BaseModel):
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: Optional[int] = None  # TEMPORARY: not yet conditionally handled
    persistent_workers: Optional[bool] = None  # TEMPORARY: not yet adapted

    _note: str = PrivateAttr(default="Dataloader fine-tuning (e.g., text bucketing, audio collate_fn) not yet implemented")


class DataIOConfig(BaseModel):
    dataset: DatasetConfig
    dataloader: DataloaderConfig
