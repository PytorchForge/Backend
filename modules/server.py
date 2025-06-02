import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import parser  

app = FastAPI()

class TrainingConfig(BaseModel):
    config: dict

class InferenceConfig(BaseModel):
    file_path: str
    model_path: str

@app.post("/train/")
async def train(config: TrainingConfig):
    generated_code = parser.generate_code(config.config)
    # write code to file
    os.makedirs("output", exist_ok=True)
    with open("output/generated_code.py", "w") as f:
        f.write(generated_code)
    return PlainTextResponse(content=generated_code)

@app.post("/inference/")
async def inference(config: InferenceConfig):
    result = parser.run_inference(
        file_path=config.file_path,
        model_path=config.model_path
    )
    return JSONResponse(content={"result": result})

