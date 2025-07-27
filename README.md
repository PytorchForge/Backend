# Backend For Pipelines

- generate code for ML tasks using templates
  - tasks
  - data processing
  - model definition
  - training configuration
- communicate with UI using FastAPI
  - receive config
  - send back generated code
  - send training command, inference results
- run training or inference scripts and return result to UI

## Running the server

- install uv with `brew install uv` or `python -m pip install uv`
- sync dependencies `uv sync` from root folder of repo
- activate venv `source .venv/bin/activate` (/activate.<shell_ext> if present in .venv/bin)
- run server `python modules/main.py`
- check api at `http://127.0.0.1:8000/docs`


## TODO: Build into a cross platform binary
