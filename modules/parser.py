import json
import argparse
from pathlib import Path
import sys
import jinja2
import isort
import black
from thefuzz import process
from configs.registries import (
    optimizerreg,
    lossreg,
    metricreg,
    layerreg,
    modelreg,
)


if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
print(f"Base directory in paerser is: {BASE_DIR}", file=sys.stderr)

sys.path.insert(0, str(BASE_DIR / "modules"))
sys.path.insert(0, str(BASE_DIR / "templates"))

DEFAULT_BASE_PATH = BASE_DIR

REGISTRY_MAP = {
    "optimizers": optimizerreg.OPTIMIZER_REGISTRY,
    "losses": lossreg.LOSS_REGISTRY,
    "metrics": metricreg.METRIC_REGISTRY,
    "layers": layerreg.LAYER_REGISTRY,
    "models": modelreg.PRETRAINED_MODEL_REGISTRY,
}


def fuzzy_match(query: str, choices: list[str], threshold: int = 75) -> str:
    """
    Perform fuzzy matching on choices to resolve query.
    """
    matched, score = process.extractOne(query, choices)
    if score >= threshold:
        return matched
    raise ValueError(f"No suitable match for '{query}'. Closest match was '{matched}' with a score of {score}.")


def setup_environment():
    base = Path(__file__).parent.parent
    if getattr(sys, 'frozen', False):
        # If running as a bundled app, use the _MEIPASS directory
        base = Path(sys._MEIPASS)
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

DEFAULT_OUTPUT = BASE_DIR / "outputs" / "out.py"


def read_config(path):
    return json.loads(Path(path).read_text())


def auto_resolve_name(category: str, provided_name: str) -> str:
    registry = REGISTRY_MAP.get(category)
    if not registry:
        raise ValueError(f"No registry found for category '{category}'.")
    return fuzzy_match(provided_name, list(registry.keys()))


def normalize_names(cfg: dict):
    training = cfg.get("training", {})
    if optimizer := training.get("optimizer"):
        optimizer["name"] = auto_resolve_name("optimizers", optimizer["name"])
    if loss := training.get("loss"):
        loss["name"] = auto_resolve_name("losses", loss["name"])
    for metric in training.get("metrics", []):
        metric["name"] = auto_resolve_name("metrics", metric["name"])
    
    model_cfg = cfg.get("model", {})
    if model_cfg.get("use_pretrained", False):
        if pretrained := model_cfg.get("pretrained", {}):
            pretrained["name"] = auto_resolve_name("models", pretrained["name"])
    elif layers := model_cfg.get("layers", []):
        for layer in layers:
            layer["type"] = auto_resolve_name("layers", layer["type"])



def resolve_template(section, cfg):
    debug_log(f"resolve_template called with section={section}")
    task_available = ["audio", "video", "image", "text", "tabular"]
    subtask_available = ["classification", "regression", "generation"]
    task_ui = cfg["main_task"].split()[0]
    subtask_ui = cfg["sub_task"]
    task_dir = fuzzy_match(task_ui, task_available)
    subtask_dir = fuzzy_match(subtask_ui, subtask_available)
    if section == "models":
        if cfg.get("model", {}).get("use_pretrained", False):
            return TASK_MAP["models"].format(task=task_dir)
        else:
            return TASK_MAP["custom_model"]
    if section == "loaders":
        return TASK_MAP[section].format(task=task_dir, subtask=subtask_dir)
    return TASK_MAP[section].format(task=task_dir)


def render_template(path, context):
    try:
        template = env.get_template(path)
        return template.render(
            config=context,
            model=context.get("model", {}),
            layers=context.get("model", {}).get("layers", []),
            METRIC_REGISTRY=metricreg.METRIC_REGISTRY,
            OPTIMIZER_REGISTRY=optimizerreg.OPTIMIZER_REGISTRY,
            LOSS_REGISTRY=lossreg.LOSS_REGISTRY,
            LAYER_REGISTRY=layerreg.LAYER_REGISTRY
        )
    except jinja2.exceptions.TemplateNotFound:
        raise ValueError(f"Template {path} not found.")


def assemble(cfg):
    normalize_names(cfg)
    parts = []
    imports_template = resolve_template("imports", cfg)
    parts.append(render_template(imports_template, cfg))
    model_template = resolve_template("models", cfg)
    parts.append(render_template(model_template, cfg))
    transforms_template = resolve_template("transforms", cfg)
    parts.append(render_template(transforms_template, cfg).strip())
    if dataloading_code := cfg.get("dataloading"):
        parts.append(dataloading_code.strip())
    static_templates = [
        "setup.j2",
        "train/utils.j2",
        "train/train_loop.j2",
        "train/eval_loop.j2",
        "train/monitoring.j2",
        "train/optimizers.j2",
        "train/losses.j2",
        "train/metrics.j2",
        "runner.j2",
    ]
    for template in static_templates:
        debug_log(f"Rendering static template: {template}")
        parts.append(env.get_template(template).render(
            config=cfg,
            METRIC_REGISTRY=metricreg.METRIC_REGISTRY,
            OPTIMIZER_REGISTRY=optimizerreg.OPTIMIZER_REGISTRY,
            LOSS_REGISTRY=lossreg.LOSS_REGISTRY,
            LAYER_REGISTRY=layerreg.LAYER_REGISTRY
        ))
    return "\n\n".join(parts)


def write_output(code: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        code = isort.code(code)
        mode = black.FileMode(line_length=88)
        code = black.format_str(code, mode=mode)
    except black.parsing.InvalidInput as e:
        debug_log(f"Black formatting error: {e}")
        raise
    output_path.write_text(code)
    # print(json.dumps({"generated_path": str(output_path.resolve())}))


def get_output_path(user_path: str = None) -> Path:
    path = Path(user_path) if user_path else DEFAULT_OUTPUT
    if path.exists():
        idx = 1
        while (candidate := path.with_name(f"{path.stem}_{idx}{path.suffix}")).exists():
            idx += 1
        return candidate
    return path


def debug_log(message: str):
    """Write debug information to stderr."""
    print(message, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Generate code from templates.")
    parser.add_argument("config", help="Path to config JSON")
    parser.add_argument("--output", help="Optional output path for generated code")
    args = parser.parse_args()
    cfg = read_config(args.config)
    code = assemble(cfg)
    out_path = get_output_path(args.output)

    # TODO: change this to use the UI code instead of saving to a file
    write_output(code, out_path)
    print(json.dumps({"generated_path": str(out_path.resolve())}))  # required for writing


if __name__ == "__main__":
    main()