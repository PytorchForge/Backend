from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any
from .registries.preprocessingreg import AUDIO_TRANSFORMS, IMAGE_TRANSFORMS, TEXT_TRANSFORMS

TRANSFORM_REGISTRY = {
    **AUDIO_TRANSFORMS,
    **IMAGE_TRANSFORMS,
    **TEXT_TRANSFORMS,
}

class PreprocessingStep(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    def apply_defaults(cls, v, info):
        if v:
            return v
        name = info.data.get("name")
        return TRANSFORM_REGISTRY.get(name, {})
