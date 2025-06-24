from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from subprocess import run as subprocess_run, PIPE
from pathlib import Path
import tempfile
import json
import os
import sys
from typing import List, Literal, Optional, Dict, Any


app = FastAPI()

DEFAULT_BASE_PATH = Path(__file__).parent # current directory of this script




class PreprocessingStep(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class LayerDefinition(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PretrainedModel(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    use_pretrained: bool
    pretrained: Optional[PretrainedModel] = None
    layers: Optional[List[LayerDefinition]] = None


class OptimizerConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class LossConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class MetricConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class SchedulerConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class EarlyStoppingConfig(BaseModel):
    enabled: bool
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int
    weight_decay: Optional[float] = 0.0
    early_stopping: Optional[EarlyStoppingConfig] = None
    scheduler: Optional[SchedulerConfig] = None


class MonitoringConfig(BaseModel):
    monitor: List[str] = Field(default_factory=list)  # e.g., ['val_loss', 'accuracy']


class DataConfig(BaseModel):
    task_type: Literal["ml", "dl"]
    main_task: str
    sub_task: str
    data_format: Literal["imagefolder", "csv", "json", "other"]
    data_type: Literal["file", "folder"]
    preprocessing: List[PreprocessingStep]


class GenerateDataModel(BaseModel):
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    monitoring: MonitoringConfig
    optimizers: List[OptimizerConfig]
    losses: List[LossConfig]
    metrics: List[MetricConfig]


class GeneratePayload(BaseModel):
    data: GenerateDataModel
    base_path: str = str(DEFAULT_BASE_PATH)


@app.post("/generate")
async def generate(payload: GeneratePayload):
    base_path = Path(payload.base_path)
    parser_path = base_path / "parser.py"
    if not parser_path.is_file():
        return JSONResponse(status_code=400, content={"error": "parser.py not found"})

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_config:
        json.dump(payload.data, temp_config)
        temp_config.flush()
        temp_config_path = temp_config.name

    try:
        result = subprocess_run(
            [sys.executable, str(parser_path), temp_config_path],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            cwd=base_path
        )
        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": result.stderr})

        parser_output = json.loads(result.stdout.strip())
        generated_path = parser_output["generated_path"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.unlink(temp_config_path)

    return FileResponse(generated_path, filename=Path(generated_path).name)



class RunPayload(BaseModel):
    base_path: str | None = None  # Optional, defaults to current directory

@app.post("/run")
async def run(payload: RunPayload):
    base_path = Path(payload.base_path) if payload.base_path else DEFAULT_BASE_PATH
    generated_path = base_path.parent / "outputs" / "out.py"
    result_json_path = base_path.parent / "outputs" / "results.json"

    if not generated_path.is_file():
        return JSONResponse(status_code=500, content={"error": "Generated file not found"})

    try:
        exec_result = subprocess_run(
            [sys.executable, str(generated_path)],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            cwd=generated_path.parent
        )
        if exec_result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": exec_result.stderr})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Execution failed: {str(e)}"})

    if not result_json_path.is_file():
        return JSONResponse(status_code=500, content={"error": "results.json not found"})

    with open(result_json_path, "r") as f:
        results = json.load(f)

    return JSONResponse(content=results)
