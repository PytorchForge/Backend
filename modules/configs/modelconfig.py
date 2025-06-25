from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any, Optional, List
from .registries.layerreg import LAYER_REGISTRY
from .registries.modelreg import PRETRAINED_MODEL_REGISTRY

class LayerDefinition(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    def apply_layer_defaults(cls, v, info):
        if v:
            return v
        layer_type = info.data.get("type")
        return LAYER_REGISTRY.get(layer_type, {})

class PretrainedModel(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    def apply_model_defaults(cls, v, info):
        if v:
            return v
        model_name = info.data.get("name")
        return PRETRAINED_MODEL_REGISTRY.get(model_name, {})

class ModelConfig(BaseModel):
    use_pretrained: bool
    pretrained: Optional[PretrainedModel] = None
    layers: Optional[List[LayerDefinition]] = None

    @model_validator(mode="before")
    @classmethod
    def check_exclusivity(cls, values):
        use_pretrained = values.get("use_pretrained")
        pretrained = values.get("pretrained")
        layers = values.get("layers")

        if use_pretrained:
            if not pretrained:
                raise ValueError("use_pretrained=True requires a 'pretrained' model.")
            if layers:
                raise ValueError("If 'use_pretrained' is True, 'layers' must be None.")
        else:
            if not layers:
                raise ValueError("use_pretrained=False requires 'layers' to be defined.")
            if pretrained:
                raise ValueError("If 'use_pretrained' is False, 'pretrained' must be None.")
        return values
