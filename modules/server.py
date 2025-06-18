from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from subprocess import run as subprocess_run, PIPE
from pathlib import Path
import tempfile
import json
import os
import sys

app = FastAPI()

DEFAULT_BASE_PATH = Path(__file__).parent # current directory of this script

class GeneratePayload(BaseModel):
    data: dict
    base_path: str = DEFAULT_BASE_PATH

class RunPayload(BaseModel):
    base_path: str | None = None  # Optional, defaults to current directory


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
