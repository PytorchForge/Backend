import json
import argparse
from pathlib import Path
import jinja2
import isort
import black

from configs.registries import (
    optimizerreg,
    lossreg,
    metricreg,
    layerreg,
    modelreg,
    aliasreg
)
from configs.registries.preprocessingreg import (
    AUDIO_TRANSFORMS,
    IMAGE_TRANSFORMS,
    TEXT_TRANSFORMS
)

REGISTRY_MAP = {
    "optimizers": optimizerreg.OPTIMIZER_REGISTRY,
    "losses": lossreg.LOSS_REGISTRY,
    "metrics": metricreg.METRIC_REGISTRY,
    "layers": layerreg.LAYER_REGISTRY,
    "models": modelreg.PRETRAINED_MODEL_REGISTRY,
}

def setup_environment():
    base = Path(__file__).parent.parent
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(base / "templates")),
        keep_trailing_newline=True,
        autoescape=False
    )

env = setup_environment()

TASK_MAP = {
    "imports": "imports.j2",
    "models": "models/{task}.j2",
    "transforms": "data/transforms/{task}.j2",
    "loaders": "data/loaders/{task}/{subtask}.j2",
    "custom_model": "models/layers.j2",
    "optimizer": "train/optimizers.j2",
    "loss": "train/losses.j2",
    "metrics": "train/metrics.j2"
}

DEFAULT_OUTPUT = Path(__file__).parent.parent / "outputs" / "out.py"

def resolve_backend_name(category: str, ui_name: str):
    return aliasreg.UI_ALIASES.get(category, {}).get(ui_name.lower(), ui_name)

def inject_registry_defaults(category: str, backend_name: str, current_params: dict):
    defaults = REGISTRY_MAP.get(category, {}).get(backend_name, {})
    for param, default in defaults.items():
        current_params.setdefault(param, default)

def normalize_training_config(training_config: dict):
    for key in ["optimizer", "loss"]:
        if training_config.get(key):
            current = training_config[key]
            backend_name = resolve_backend_name(f"{key}s", current["name"])
            current["name"] = backend_name
            current.setdefault("params", {})
            inject_registry_defaults(f"{key}s", backend_name, current["params"])

    for metric in training_config.get("metrics", []):
        backend_metric_name = resolve_backend_name("metrics", metric["name"])
        metric["name"] = backend_metric_name
        metric.setdefault("params", {})
        inject_registry_defaults("metrics", backend_metric_name, metric["params"])

def normalize_model_config(model_cfg: dict):
    if model_cfg.get("use_pretrained") and model_cfg.get("pretrained"):
        ui_name = model_cfg["pretrained"]["name"]
        backend_name = resolve_backend_name("models", ui_name)
        model_cfg["pretrained"]["name"] = backend_name
        model_cfg["pretrained"].setdefault("params", {})
        inject_registry_defaults("models", backend_name, model_cfg["pretrained"]["params"])

    elif model_cfg.get("layers"):
        for layer in model_cfg["layers"]:
            ui_name = layer["type"]
            backend_name = resolve_backend_name("layers", ui_name)
            layer["type"] = backend_name
            layer.setdefault("params", {})
            inject_registry_defaults("layers", backend_name, layer["params"])

def normalize_data_config(data_cfg: dict):
    sub_task_ui = data_cfg.get("sub_task")
    data_format_ui = data_cfg.get("data_format")

    data_cfg["sub_task"] = resolve_backend_name("subtasks", sub_task_ui)
    data_cfg["data_format"] = resolve_backend_name("data_formats", data_format_ui)

def normalize_preprocessing(preprocessing_steps: list, main_task: str):
    task_key = main_task.strip().lower()

    if task_key in ["audio processing", "audio"]:
        registry = AUDIO_TRANSFORMS
    elif task_key in ["text processing", "text"]:
        registry = TEXT_TRANSFORMS
    elif task_key in ["image processing", "image"]:
        registry = IMAGE_TRANSFORMS
    else:
        raise ValueError(f"No transforms registry found for main_task: {main_task}")

    for step in preprocessing_steps:
        ui_name = step["name"]
        backend_name = resolve_backend_name("transforms", ui_name)
        step["name"] = backend_name
        step.setdefault("params", {})
        inject_defaults_from_registry(step["params"], registry, backend_name)

def inject_defaults_from_registry(params: dict, registry: dict, backend_name: str):
    defaults = registry.get(backend_name)
    if defaults is None:
        raise ValueError(f"No defaults found in registry for transform: {backend_name}")
    for param, default in defaults.items():
        params.setdefault(param, default)

def normalize_config(cfg: dict):
    main_task = cfg.get("data", {}).get("main_task", "")
    normalize_data_config(cfg.get("data", {}))
    normalize_training_config(cfg.get("training", {}))
    normalize_model_config(cfg.get("model", {}))
    
    if preprocessing_cfg := cfg.get("preprocessing"):
        normalize_preprocessing(preprocessing_cfg, main_task)

def read_config(path):
    return json.loads(Path(path).read_text())

def resolve_template(section, cfg):
    task_dir = cfg["data"]["main_task"].lower().split()[0]
    subtask_dir = cfg["data"]["sub_task"].lower()

    subtask_dir = resolve_backend_name("subtasks", subtask_dir)

    if section == "models" and cfg.get("model", {}).get("modelType") == "custom":
        return TASK_MAP["custom_model"]
    
    if section == "loaders":
        return TASK_MAP[section].format(task=task_dir, subtask=subtask_dir)

    return TASK_MAP[section].format(task=task_dir)

def render_template(name, context):
    try:
        tpl = env.get_template(name)
    except jinja2.exceptions.TemplateNotFound:
        raise ValueError(f"Template not found: {name}")
    return tpl.render(config=context)

def assemble(cfg):
    normalize_config(cfg)

    parts = []

    imports_template = resolve_template("imports", cfg)
    parts.append(render_template(imports_template, cfg))

    for sec in ["models", "transforms"]:
        template_name = resolve_template(sec, cfg)
        parts.append(render_template(template_name, cfg))

    # use user code instead of template for loaders
    dataloading_code = cfg.get("dataloading", "")
    if dataloading_code:
        parts.append("# Explicit dataloading code provided by UI:\n\n")
        parts.append(dataloading_code.strip())

    static_templates = ["setup.j2", "train/utils.j2", "train/train_loop.j2", "train/eval_loop.j2"]
    for template in static_templates:
        if template in env.list_templates():
            parts.append(env.get_template(template).render(config=cfg))

    if "monitoring.j2" in env.list_templates():
        parts.append(env.get_template("train/monitoring.j2").render(config=cfg))

    if "runner.j2" in env.list_templates():
        parts.append(env.get_template("runner.j2").render(config=cfg))

    return "\n\n".join(parts)

def write_output(code: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    code = isort.code(code)
    mode = black.FileMode(line_length=88)
    code = black.format_str(code, mode=mode)
    output_path.write_text(code)

def get_output_path(user_path: str = None) -> Path:
    if user_path:
        return Path(user_path)
    path = DEFAULT_OUTPUT
    if path.exists():
        idx = 1
        while True:
            candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
            if not candidate.exists():
                return candidate
            idx += 1
    return path

def main():
    parser = argparse.ArgumentParser(description="Generate code from templates.")
    parser.add_argument("config", help="Path to config JSON")
    parser.add_argument("--output", help="Optional output path for generated code")
    args = parser.parse_args()

    cfg = read_config(args.config)
    code = assemble(cfg)
    out_path = get_output_path(args.output)
    write_output(code, out_path)

    results = {"generated_path": str(out_path.resolve())}
    print(json.dumps(results))

if __name__ == "__main__":
    main()