from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from .registries.lossreg import LOSS_REGISTRY
from .registries.metricreg import METRIC_REGISTRY
from .registries.optimizerreg import OPTIMIZER_REGISTRY
from .registries.schedulerreg import SCHEDULER_REGISTRY

MonitoringTool = Literal[
    "use_tensorboard", 
    "use_wandb", 
    "use_mlflow", 
    "resource_alerts", 
    "threshold_alerts"
]

class OptimizerConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    def apply_optimizer_defaults(cls, v, info):
        return v or OPTIMIZER_REGISTRY.get(info.data["name"], {})


class SchedulerConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    def apply_scheduler_defaults(cls, v, info):
        return v or SCHEDULER_REGISTRY.get(info.data["name"], {})


class LossConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    def apply_loss_defaults(cls, v, info):
        return v or LOSS_REGISTRY.get(info.data["name"], {})


class MetricConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    def apply_metric_defaults(cls, v, info):
        return v or METRIC_REGISTRY.get(info.data["name"], {})


class EarlyStoppingConfig(BaseModel):
    enabled: bool
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    def apply_earlystop_defaults(cls, v, info):
        return v or {"monitor": "val_loss", "patience": 10, "mode": "min"}

class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int
    weight_decay: Optional[float] = Field(default=0.0)

    optimizer: Optional[OptimizerConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    loss: Optional[LossConfig] = None
    metrics: Optional[List[MetricConfig]] = None
    early_stopping: Optional[EarlyStoppingConfig] = None
    monitoring: Optional[List[MonitoringTool]] = Field(default_factory=list)
    @field_validator("optimizer", "scheduler", "loss", "metrics", "early_stopping","monitoring", mode="before")
    def inject_defaults(cls, v):
        return v or None
