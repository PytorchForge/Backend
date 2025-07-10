from pydantic import BaseModel, PrivateAttr
from typing import Optional, Dict, Literal, Any

TaskType = Literal["Image Processing", "Text Processing", "Audio Processing"]
SubTaskType = Literal[
    # Image
    "Image Classification", "Object Detection", "Image Segmentation", "Image Generation",
    # Text
    "Text Classification", "Text Generation", "Machine Translation", "Text Summarization",
    # Audio
    "Speech Recognition", "Audio Classification", "Audio Generation", "Voice Conversion"
]

class DatasetConfig(BaseModel):
    split_type: Literal["include", "exclude"] = "include"
    label_type: Literal["folder-name", "file", "csv", "none"] = "folder-name"
    label_map: Optional[Dict[str, int]] = None
    return_format: Literal["raw", "dict", "tuple"] = "dict"

    # Text-specific
    tokenizer_type: Optional[str] = None  
    tokenizer_params: Optional[Dict[str, Any]] = None  

    # Audio-specific
    audio_duration: Optional[float] = None 

    # Task flags
    include_prompt: Optional[bool] = None
    include_target: Optional[bool] = None
    text_pair: Optional[bool] = None
    audio_pair: Optional[bool] = None
    multi_class: Optional[bool] = None
    multi_label: Optional[bool] = None
    binary: Optional[bool] = None

    _note: str = PrivateAttr(default="Tokenization/audio-specific logic not yet fully defined")

class DataloaderConfig(BaseModel):
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: Optional[int] = None  # TEMPORARY
    persistent_workers: Optional[bool] = None  # TEMPORARY

    _note: str = PrivateAttr(default="Dataloader fine-tuning (e.g., bucketing, custom collate) not yet implemented")

class DataIOConfig(BaseModel):
    task: TaskType
    subtask: SubTaskType
    dataset: DatasetConfig
    dataloader: DataloaderConfig
