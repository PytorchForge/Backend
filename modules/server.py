from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from modules.parser import generate_code
import os

app = FastAPI()

@app.post("/generate/")
async def generate_endpoint(request: Request):
    config = await request.json()
    output_path = "output/generated_model.py"
    try:
        generate_code(config, output_path)
        return JSONResponse({"status": "success", "output_file": output_path})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)
