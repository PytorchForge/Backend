import json
import argparse
from pathlib import Path
import jinja2
import isort
import black

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
}

DEFAULT_OUTPUT = Path(__file__).parent.parent / "outputs" / "out.py"

def read_config(path):
    return json.loads(Path(path).read_text())

def resolve_name(section, cfg):
    if section == "models" and cfg.get("modelType") == "custom":
        return TASK_MAP["custom_model"]
    return TASK_MAP[section].format(
        task=cfg["mainTask"].lower(),
        subtask=cfg["subTask"].lower()
    )

def render_template(name, context):
    tpl = env.get_template(name)
    return tpl.render(config=context)

def assemble(cfg):
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

if __name__ == "__main__":
    main()