import uuid
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from subprocess import run as subprocess_run, PIPE
from pathlib import Path
import tempfile
import json
import os
import sys
from typing import Any, Dict, List, Literal, Optional
import ast
import toml

# from configs.datasetconfig import DataIOConfig
# from configs.taskconfig import DataFormat, DataType
# from configs.preprocessingconfig import PreprocessingStep
# from configs.modelconfig import ModelConfig
# from configs.trainconfig import TrainingConfig


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent

DEFAULT_BASE_PATH = BASE_DIR
print(f"Default base path set to: {DEFAULT_BASE_PATH}")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs",status_code=308)

@app.get("/health", include_in_schema=False)
async def health_check():
    return JSONResponse(status_code=200, content={"status": "ok", "message": "Server is running"})

class DataSource(BaseModel):
    type: Literal["file", "folder"]
    value: str  # Path to file or folder

class PreprocessingStep(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class PretrainedModel(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class LayerDefinition(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)

class ModelConfig(BaseModel):
    use_pretrained: bool
    pretrained: Optional[PretrainedModel] = None
    layers: Optional[List[LayerDefinition]] = None

class LossConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class OptimizerConfig(BaseModel):
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

class HyperParams(BaseModel):
    batch_size: int
    learning_rate: float
    epochs: int
    weight_decay: Optional[float] = 0.0

class TrainingConfig(BaseModel):
    loss: LossConfig
    optimizer: OptimizerConfig
    metrics: List[MetricConfig]
    scheduler: Optional[SchedulerConfig] = None
    monitoring: List[Literal["use_tensorboard", "use_wandb", "use_mlflow"]] = Field(default_factory=list)
    hyper_params: HyperParams
    early_stopping: EarlyStoppingConfig

class GeneratePayload(BaseModel):
    task_type: Literal["ml", "dl"]
    main_task: str
    sub_task: str
    data_format: str = Field(..., description="Format of the data, e.g., 'csv', 'json', 'image', 'text'")
    data_source: DataSource
    dataloading: str
    preprocessing: List[PreprocessingStep] = Field(default_factory=list)
    model: ModelConfig
    training: TrainingConfig
    base_path: Optional[str] = DEFAULT_BASE_PATH.as_posix() 




DATASETS_ROOT = DEFAULT_BASE_PATH / Path("/dataset")


def parse_imports_from_code(code: str):
    """Extract imported packages from Python code."""
    imported_packages = set()
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_packages.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_packages.add(node.module.split('.')[0])
    return imported_packages

# Map imported module names to PyPI package names 
IMPORT_TO_PACKAGE_MAP = {
    "PIL": "pillow",
    "sklearn": "scikit-learn",

}
def update_pyproject_toml(pyproject_path: Path, packages: set):
    if not pyproject_path.is_file():
        # Create minimal PEP 621-compliant structure
        pyproject_data = {
            "project": {
                "dependencies": []
            }
        }
    else:
        pyproject_data = toml.load(pyproject_path)

    dependencies = set(pyproject_data.get("project", {}).get("dependencies", []))
    
    # ignore built-in
    builtin_packages = {"sys", "os", "json", "logging", "pathlib", "shutil", "tempfile","random", "subprocess", "ast", "typing"}
    packages -= builtin_packages

    resolved_packages = set()
    for pkg in packages:
        resolved_pkg = IMPORT_TO_PACKAGE_MAP.get(pkg, pkg)
        resolved_packages.add(resolved_pkg)

    existing_dep_names = {dep.split(">=")[0].lower() for dep in dependencies}
    new_packages = {f"{pkg}>=0" for pkg in resolved_packages if pkg.lower() not in existing_dep_names}

    if new_packages:
        dependencies.update(new_packages)
        pyproject_data["project"]["dependencies"] = sorted(dependencies)
        pyproject_path.parent.mkdir(parents=True, exist_ok=True)
        toml.dump(pyproject_data, pyproject_path.open("w"))

@app.post("/generate")
async def generate(payload: GeneratePayload):
    base_path = Path(payload.base_path) if payload.base_path else DEFAULT_BASE_PATH
    parser_path = (BASE_DIR /"modules" /"parser.py").resolve()


    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_config:
        cfg_dict = payload.model_dump()
        cfg_dict["dataloading"] = payload.dataloading.strip()
        json.dump(cfg_dict, temp_config)
        temp_config.flush()
        temp_config_path = temp_config.name

    try:
        python_exec = "python3" if getattr(sys, "frozen", False) else sys.executable

        result = subprocess_run(
            [python_exec, str(parser_path), temp_config_path],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            cwd=base_path,
        )
        
        print("Subprocess STDOUT:")
        print(result.stdout)
        print("Subprocess STDERR:")
        print(result.stderr)

        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": result.stderr.strip()})

        if not result.stdout.strip():
            return JSONResponse(status_code=500, content={"error": "Parser returned empty output"})
        
        try:
            parser_output = json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            print(f"Failed to parse output: {result.stdout.strip()}")
            return JSONResponse(status_code=500, content={"error": f"Failed to parse output: {str(e)}"})
        
        generated_path = Path(parser_output["generated_path"])

        if not generated_path.exists():
            return JSONResponse(status_code=404, content={"error": f"{generated_path} not found"})
        
        generated_code = generated_path.read_text()

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.unlink(temp_config_path)

    return JSONResponse(
        status_code=200,
        content={"generated_code": generated_code, "generated_path": str(generated_path)},
    )                         


class RunPayload(BaseModel):
    base_path: Optional[str] | None = None  #defaults to current directory
    generated_code: str = Field(..., description="code to run")

@app.post("/run")
async def run(payload: RunPayload):
    base_path = Path(payload.base_path) if payload.base_path and payload.base_path.lower() != "string" else DEFAULT_BASE_PATH
    outputs_path = base_path if base_path.name == "outputs" else base_path.parent / "outputs"

    if not outputs_path.is_dir():
        return JSONResponse(status_code=500, content={"error": f"Outputs directory '{outputs_path}' not found."})

    temp_code_path = outputs_path / f"temp_{uuid.uuid4().hex}.py"
    temp_code_path.write_text(payload.generated_code)

    result_json_path = outputs_path / "results.json"

    venv_path = outputs_path / ".venv"
    pip = venv_path / "bin" / "pip"
    uv = venv_path / "bin" / "uv"
    python_exec = venv_path / "bin" / "python"

    try:
        try:
            subprocess_run([pip, "install", "uv"], cwd=outputs_path, stderr=PIPE, stdout=PIPE, check=True)
        except Exception:
            subprocess_run(["brew", "install", "uv"], cwd=outputs_path, stderr=PIPE, stdout=PIPE, check=True)
        except Exception:
            raise RuntimeError("Failed to install uv. install pip or brew first.")

        try:
            subprocess_run([uv, "sync"], cwd=outputs_path, stderr=PIPE, stdout=PIPE, check=True)
        except Exception:
            raise RuntimeError("Failed to sync uv environment. Make sure uv is installed correctly.")

        exec_result = subprocess_run(
            [python_exec, str(temp_code_path)],
            stdout=PIPE,
            stderr=PIPE,
            cwd=outputs_path,
            text=True
        )
        if exec_result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": exec_result.stderr})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Execution failed: {str(e)}"})

    finally:
        if temp_code_path.is_file():
            temp_code_path.unlink()  

    if not result_json_path.is_file():
        return JSONResponse(status_code=500, content={"error": "results.json not found"})

    results = json.loads(result_json_path.read_text())
    return JSONResponse(content=results)



# class InferPayload(BaseModel):
#     base_path: Optional[str] = None
#     task: Literal["image", "text", "audio"] = Field(..., description="Type of task for inference")
#     subtask: Literal[
#         "Image Classification", "Object Detection", "Image Segmentation", "Image Generation",
#         "Text Classification", "Text Generation", "Machine Translation", "Text Summarization",
#         "Speech Recognition", "Audio Classification", "Audio Generation", "Voice Conversion"
#     ] = Field(..., description="Subtask for inference")
    
#     model_path: str = Field(..., description="Path to the model file")
#     model_load_method: Literal["torch.load", "onnx"] = Field(..., description="How to load the model")

#     input_data: List[str] = Field(..., description="List of input data for inference")
#     input_size: Optional[int] = None
#     output_type: Literal["json", "text", "image", "audio", "multitype"] = "json"

#     tokenizer_type: Optional[str] = None
#     tokenizer_params: Optional[Dict[str, Any]] = None
#     return_logits: Optional[bool] = False
#     return_probs: Optional[bool] = False
#     top_k: Optional[int] = None
#     temperature: Optional[float] = None
#     max_length: Optional[int] = None

#     _note: Optional[str] = "Inference pipeline defaults may not cover all edge cases yet"

#     @model_validator(mode="after")
#     def set_task_defaults(self) -> "InferPayload":
#         if self.task == "text":
#             if self.subtask in {"Text Generation", "Machine Translation", "Text Summarization"}:
#                 if not self.tokenizer_type:
#                     raise ValueError("tokenizer_type must be set for text generation tasks")
#                 if self.max_length is None:
#                     self.max_length = 128
#                 if self.temperature is None:
#                     self.temperature = 1.0

#         if self.task == "audio":
#             if self.subtask in {"Speech Recognition", "Audio Generation", "Voice Conversion"}:
#                 if self.input_size is None:
#                     self.input_size = 16000

#         if self.task == "image" and self.input_size is None:
#             self.input_size = 224

#         return self

# @app.post("/generate_inference")
# async def generate_inference(payload: InferPayload):
#     base_path = Path(payload.base_path) if payload.base_path else DEFAULT_BASE_PATH
#     infer_script_path = base_path / "outputs" /"infer.py"
#     if not infer_script_path.is_file():
#         return JSONResponse(status_code=400, content={"error": "infer.py not found"})

#     try:
#         result = subprocess_run(
#             [sys.executable, str(infer_script_path)],
#             input=payload.json(),
#             stdout=PIPE,
#             stderr=PIPE,
#             text=True,
#             cwd=base_path
#         )
#         if result.returncode != 0:
#             return JSONResponse(status_code=500, content={"error": result.stderr})

#         output = json.loads(result.stdout.strip())
#         return JSONResponse(content=output)
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# @app.post("/run_inference")
# async def inference(payload: InferPayload):
#     base_path = Path(payload.base_path) if payload.base_path else DEFAULT_BASE_PATH
#     infer_script_path = base_path / "infer.py"
#     if not infer_script_path.is_file():
#         return JSONResponse(status_code=400, content={"error": "infer.py not found"})
#     try:
#         result = subprocess_run(
#             [sys.executable, str(infer_script_path)],
#             input=payload.json(),
#             stdout=PIPE,
#             stderr=PIPE,
#             text=True,
#             cwd=base_path
#         )
#         if result.returncode != 0:
#             return JSONResponse(status_code=500, content={"error": result.stderr})

#         output = json.loads(result.stdout.strip())
#         return JSONResponse(content=output)
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
    
