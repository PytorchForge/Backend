from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, field_validator

from .registries.taskreg import DEFAULTS

# from .preprocessingconfig import PreprocessingStep

class TaskType(str, Enum):
    ml = "ml"
    dl = "dl"

class DataFormat(str, Enum):
    imagefolder = "imagefolder"
    csv = "csv"
    json = "json"
    png = "png"
    jpg = "jpg"
    jpeg = "jpeg"
    txt = "txt"
    audio = "audio"
    video = "video"
    parquet = "parquet"
    other = "other"

class DataType(str, Enum):
    file = "file"
    folder = "folder"

class TaskConfig(BaseModel):
    task_type: Literal["ml", "dl"]
    main_task: str
    sub_task: str
    data_format: Optional[DataFormat] = None
    data_type: Optional[DataType] = None
    # preprocessing: Optional[List[PreprocessingStep]] = None  #duplicate

    @field_validator("data_format", "data_type", mode="before")
    def set_defaults(cls, v, info):
        if v is not None:
            return v
        values = info.data
        if values.get("task_type") == "ml":
            return None
        sub_task = values.get("sub_task")
        if not sub_task:
            return None
        defaults = DEFAULTS.get(sub_task)
        if not defaults:
            return None
        field_name = info.field_name
        return defaults.get(field_name)