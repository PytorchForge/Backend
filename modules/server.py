from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from subprocess import run as subprocess_run, PIPE
import tempfile
import os
import sys
import json

app = FastAPI()

class InputPayload(BaseModel):
    data: dict
    base_path: str
    command: str = "generate"  # "generate" or "run generated file"

@app.post("/process")
async def process(payload: InputPayload):
    parser_path = os.path.join(payload.base_path, "parser.py")
    if not os.path.isfile(parser_path):
        return JSONResponse(status_code=400, content={"error": "parser.py not found"})

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_config:
        json.dump(payload.data, temp_config)
        temp_config.flush()
        temp_config_path = temp_config.name

    try:
        result = subprocess_run(
            [sys.executable, parser_path, temp_config_path],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            cwd=payload.base_path
        )
        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": result.stderr})

        parser_output = json.loads(result.stdout.strip())
        generated_path = parser_output["generated_path"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.unlink(temp_config_path)

    if payload.command == "generate":
        return FileResponse(generated_path, filename=os.path.basename(generated_path))

    try:
        exec_result = subprocess_run(
            [sys.executable, generated_path],
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            cwd=os.path.dirname(generated_path)
        )
        if exec_result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": exec_result.stderr})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Execution failed: {str(e)}"})

    result_json_path = os.path.join(payload.base_path, "../outputs/results.json")
    if not os.path.isfile(result_json_path):
        return JSONResponse(status_code=500, content={"error": "results.json not found"})

    with open(result_json_path, "r") as f:
        results = json.load(f)

    return JSONResponse(content=results)
