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
    "metrics": "train/metrics.j2",
    "monitoring":"train/monitoring.j2"
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
        
def normalize_name(name: str, alias_map: dict) -> str:
    key = name.strip().lower()
    return alias_map.get(key, name)
    
def normalize_config(cfg):
    opt = cfg.get("optimizer")
    if opt and "name" in opt:
        opt["name"] = normalize_name(opt["name"], OPTIMIZER_ALIASES)

    loss = cfg.get("loss")
    if loss and "name" in loss:
        loss["name"] = normalize_name(loss["name"], LOSS_ALIASES)

    metric = cfg.get("metric")
    if metric and "name" in metric:
        metric["name"] = normalize_name(metric["name"], METRIC_ALIASES)
        
def read_config(path):
    return json.loads(Path(path).read_text())

def resolve_name(section, cfg):
    if section == "models" and cfg.get("modelType") == "custom":
        return TASK_MAP["custom_model"]

    #  This line fixes the "space in name" issue
    task = cfg["main_task"].strip().lower().replace(" ", "_")
    subtask = cfg["sub_task"].strip().lower().replace(" ", "_")

    resolved_path = TASK_MAP[section].format(task=task, subtask=subtask)
    print(f"[DEBUG] Looking for template: {resolved_path}")  # Optional but helpful
    return resolved_path

def render_template(name, context):
    try:
        tpl = env.get_template(name)
    except jinja2.exceptions.TemplateNotFound:
        raise ValueError(f"Template not found: {name}")
    return tpl.render(config=context)

def assemble(cfg):
    normalize_config(cfg)
    parts = []
    imp_tpl = resolve_name("imports", cfg)
    parts.append(render_template(imp_tpl, cfg) + "\n\n")
    for sec in ["models", "transforms", "loaders"]:
        tpl_name = resolve_name(sec, cfg)
        parts.append(render_template(tpl_name, cfg) + "\n\n")
    static = ["setup.j2", "train/utils.j2", "train/train_loop.j2", "train/eval_loop.j2"]
    for tpl in static:
        if tpl in env.loader.list_templates():
            parts.append(env.get_template(tpl).render(config=cfg) + "\n\n")
    for shared in ["train/optimizers.j2", "train/losses.j2", "train/metrics.j2", "train/monitoring.j2"]:
        if shared in env.loader.list_templates():
            parts.append(env.get_template(shared).render(config=cfg) + "\n\n")
    return "".join(parts)

def write_output(code: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_code = isort.code(code)
    formatted_code = black.format_str(sorted_code, mode=black.FileMode())
    output_path.write_text(formatted_code)


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