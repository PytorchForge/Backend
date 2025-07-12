# inference_parser.py
import json
import argparse
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

TEMPLATE_MAP = {
    ("text", "Text Classification"): "text_classification.j2",
    ("audio", "Audio Classification"): "audio_classification.j2",
    ("image", "Image Classification"): "image_classification.j2"
    # Add more mappings as needed
}

def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    task = config["task"]
    subtask = config["subtask"]
    template_name = TEMPLATE_MAP.get((task, subtask))

    if not template_name:
        raise ValueError(f"No template for ({task}, {subtask})")

    env = Environment(loader=FileSystemLoader("templates/inference"))
    template = env.get_template(template_name)

    rendered_code = template.render(
        tokenizer_type=config.get("tokenizer_type", "AutoTokenizer"),
        tokenizer_params=config.get("tokenizer_params", {}),
        model_path=config["model_path"],
        max_length=config.get("max_length", 128),
        return_logits=config.get("return_logits", False),
        return_probs=config.get("return_probs", False)
    )

    infer_py_path = Path("outputs/infer.py")
    infer_py_path.parent.mkdir(exist_ok=True, parents=True)
    with open(infer_py_path, "w") as f:
        f.write(rendered_code)

    # Also save infer config for use inside infer.py
    with open("outputs/infer_config.json", "w") as f:
        json.dump(config, f)

    print(json.dumps({"generated_path": str(infer_py_path)}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to JSON inference config")
    args = parser.parse_args()
    main(args.config)
