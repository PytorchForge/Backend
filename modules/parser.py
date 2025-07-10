import json
import argparse
import os
from pathlib import Path
import jinja2
import isort
import black

try:
    from aliases import OPTIMIZER_ALIASES, LOSS_ALIASES, METRIC_ALIASES
except ImportError:
	OPTIMIZER_ALIASES, LOSS_ALIASES, METRIC_ALIASES = {},{},{}
	
	
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
        
def resolve_template_from_alias(section: str, cfg: dict) -> str:
    if section == "optimizer":
        return normalize_name(cfg["optimizer"]["name"], OPTIMIZER_ALIASES)
    elif section == "loss":
        return normalize_name(cfg["loss"]["name"], LOSS_ALIASES)
    elif section == "metric":
        return normalize_name(cfg["metric"]["name"], METRIC_ALIASES)
    else:
        raise ValueError(f"Unsupported alias-resolved section: {section}")


UI_TO_BACKEND_NAME_MAP = {
    "vit": "ViT",
    "visiontransformer": "ViT",
    "plain-text": "csv",
    "wav": "audio",
    "mp3": "audio",
    "flac": "audio",
    "pickle": "other",
    "pytorch-tensor": "other",
    "image-segmentation": "segmentation",
    "object-detection": "object_detection"
}

def normalize_name(name: str, alias_map: dict) -> str:
    key = name.strip().lower()
    key = UI_TO_BACKEND_NAME_MAP.get(key, key)  # Add this line
    return alias_map.get(key, name)
    
def normalize_config(cfg):
    data_cfg = cfg.get("data", {})

    sub_task_ui = data_cfg.get("sub_task", "")
    cfg["data"]["sub_task"] = UI_TO_BACKEND_NAME_MAP.get(sub_task_ui.lower(), sub_task_ui)

    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    if opt_cfg and "name" in opt_cfg:
        opt_cfg["name"] = normalize_name(opt_cfg["name"], OPTIMIZER_ALIASES)

    loss_cfg = cfg.get("training", {}).get("loss", {})
    if loss_cfg := cfg.get("training", {}).get("loss"):
        loss_cfg["name"] = normalize_name(loss_cfg["name"], LOSS_ALIASES)

    metric_cfgs = cfg.get("training", {}).get("metrics", [])
    for metric_cfg in metric_cfgs:
        metric_cfg["name"] = normalize_name(metric_cfg["name"], METRIC_ALIASES)
        
def read_config(path):
    return json.loads(Path(path).read_text())


SUBTASK_MAPPING = {
    "image classification": "classification",
    "object detection": "object_detection",
    "image segmentation": "segmentation",
    "image generation": "generation",
    "text classification": "classification",
    "text generation": "generation",
    "machine translation": "translation",
    "text summarization": "summarisation",
    "speech recognition": "recognition",
    "audio classification": "classification",
    "audio generation": "generation",
    "voice conversion": "conversion"
}

def resolve_name(section, cfg):
    main_task = cfg["data"]["main_task"].lower()
    sub_task_original = cfg["data"]["sub_task"].lower()

    TASK_MAPPING = {
        'image processing': 'image',
        'text processing': 'text',
        'audio processing': 'audio',
        'ml': 'ML'
    }

    task_dir = TASK_MAPPING.get(main_task)
    if task_dir is None:
        raise ValueError(f"No task mapping found for main_task: '{main_task}'")

    subtask_dir = SUBTASK_MAPPING.get(sub_task_original)
    if subtask_dir is None and section == 'loaders':
        raise ValueError(f"No subtask mapping for sub_task: '{sub_task_original}'")

    if section == "models" and cfg.get("model", {}).get("modelType") == "custom":
        return TASK_MAP["custom_model"]

    if section == "loaders":
        return TASK_MAP[section].format(task=task_dir, subtask=subtask_dir)

    else:
        return TASK_MAP[section].format(task=task_dir, subtask='')


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

def render_template(name, context):
    try:
        tpl = env.get_template(name)
    except jinja2.exceptions.TemplateNotFound:
        raise ValueError(f"Template not found: {name}")
    return tpl.render(config=context)

def assemble(cfg):
    normalize_config(cfg)
    parts = []

    imports_template = resolve_name("imports", cfg)
    parts.append(render_template(imports_template, cfg))

    sections = ["models", "transforms", "loaders"]
    for sec in sections:
        template_name = resolve_name(sec, cfg)
        parts.append(render_template(template_name, cfg))

    static_templates = [
        "setup.j2",
        "train/utils.j2",
        "train/train_loop.j2",
        "train/eval_loop.j2"
    ]
    for template in static_templates:
        if template in env.list_templates():
            parts.append(env.get_template(template).render(config=cfg))

    # shared_components = ["train/optimizers.j2", "train/losses.j2", "train/metrics.j2"]
    # for shared in shared_components:
    #     parts.append(env.get_template(shared).render(config=cfg))

    if "monitoring" in cfg:
        parts.append(env.get_template("train/monitoring.j2").render(config=cfg))

    if "runner.j2" in env.list_templates():
        parts.append(env.get_template("runner.j2").render(config=cfg))

    combined_code = "\n\n".join(parts)
    return combined_code


def write_output(code: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # remove duplicates, organize imports 
    code = isort.code(code)
    # format code, remove extra newlines
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

    results = {"generated_path": os.path.abspath(out_path)}
    print(json.dumps(results)) 
    return results

if __name__ == "__main__":
    main()