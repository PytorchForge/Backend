from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from subprocess import run as subprocess_run, PIPE
from pathlib import Path
import tempfile
import json
import os
import sys
from typing import Any, Dict, List, Literal, Optional
import shutil


from configs.datasetconfig import DataIOConfig
from configs.taskconfig import TaskConfig
from configs.preprocessingconfig import PreprocessingStep
from configs.modelconfig import ModelConfig
from configs.trainconfig import TrainingConfig

app = FastAPI()

DEFAULT_BASE_PATH = Path(__file__).parent # current directory of this script



class GenerateDataModel(BaseModel):
    data: TaskConfig
    dataloading: DataIOConfig
    preprocessing: Optional[List[PreprocessingStep]] = Field(
        None, description="List of preprocessing steps to apply to the data"
    )
    model: ModelConfig
    training: TrainingConfig



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
        json.dump(payload.data.model_dump(), temp_config)
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

        # Explicitly copy config json AFTER parser_output clearly generated explicitly
        dest_config_path = Path(generated_path).parent / "config.json"
        shutil.copy(temp_config_path, dest_config_path)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.unlink(temp_config_path)

    return JSONResponse(status_code=200, content={"generated_path": generated_path})



class RunPayload(BaseModel):
    base_path: Optional[str] | None = None  #defaults to current directory

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



class InferPayload(BaseModel):
    base_path: Optional[str] = None
    task: Literal["image", "text", "audio"] = Field(..., description="Type of task for inference")
    subtask: Literal[
        "Image Classification", "Object Detection", "Image Segmentation", "Image Generation",
        "Text Classification", "Text Generation", "Machine Translation", "Text Summarization",
        "Speech Recognition", "Audio Classification", "Audio Generation", "Voice Conversion"
    ] = Field(..., description="Subtask for inference")
    
    model_path: str = Field(..., description="Path to the model file")
    model_load_method: Literal["torch.load", "onnx"] = Field(..., description="How to load the model")

    input_data: List[str] = Field(..., description="List of input data for inference")
    input_size: Optional[int] = None
    output_type: Literal["json", "text", "image", "audio", "multitype"] = "json"

    tokenizer_type: Optional[str] = None
    tokenizer_params: Optional[Dict[str, Any]] = None
    return_logits: Optional[bool] = False
    return_probs: Optional[bool] = False
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    max_length: Optional[int] = None

    _note: Optional[str] = "Inference pipeline defaults may not cover all edge cases yet"

    @model_validator(mode="after")
    def set_task_defaults(self) -> "InferPayload":
        if self.task == "text":
            if self.subtask in {"Text Generation", "Machine Translation", "Text Summarization"}:
                if not self.tokenizer_type:
                    raise ValueError("tokenizer_type must be set for text generation tasks")
                if self.max_length is None:
                    self.max_length = 128
                if self.temperature is None:
                    self.temperature = 1.0

        if self.task == "audio":
            if self.subtask in {"Speech Recognition", "Audio Generation", "Voice Conversion"}:
                if self.input_size is None:
                    self.input_size = 16000

        if self.task == "image" and self.input_size is None:
            self.input_size = 224

        return self

@app.post("/generate_inference")
async def generate_inference(payload: InferPayload):
    base_path = Path(payload.base_path) if payload.base_path else DEFAULT_BASE_PATH
    infer_script_path = base_path / "outputs" /"infer.py"
    if not infer_script_path.is_file():
        return JSONResponse(status_code=400, content={"error": "infer.py not found"})

    try:
        result = subprocess_run(
            [sys.executable, str(infer_script_path)],
            input=payload.json(),
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            cwd=base_path
        )
        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": result.stderr})

        output = json.loads(result.stdout.strip())
        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/run_inference")
async def inference(payload: InferPayload):
    base_path = Path(payload.base_path) if payload.base_path else DEFAULT_BASE_PATH
    infer_script_path = base_path / "infer.py"
    if not infer_script_path.is_file():
        return JSONResponse(status_code=400, content={"error": "infer.py not found"})
    try:
        result = subprocess_run(
            [sys.executable, str(infer_script_path)],
            input=payload.json(),
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            cwd=base_path
        )
        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": result.stderr})

        output = json.loads(result.stdout.strip())
        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    