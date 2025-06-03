import os
from jinja2 import Environment, FileSystemLoader

# placeholder for now
TEMPLATE_DIR = "templates"
OUTPUT_DIR = "output"

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), trim_blocks=True, lstrip_blocks=True)

def generate_code(config: dict, output_path: str = os.path.join(OUTPUT_DIR, "generated_model.py")):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    context = {}

    context["preprocessing"] = config.get("preprocessing", [])
    context["training"] = config.get("training", {})
    context["optimizer"] = config.get("optimizer", {})
    context["loss"] = config.get("loss", {})
    context["metrics"] = config.get("metrics", {})
    context["monitoring"] = config.get("monitoring", {})
    context["fileMetadata"] = config.get("fileMetadata", {})

    model_type = config.get("modelType")
    if model_type == "custom":
        context["layers"] = config.get("customLayers", [])
        context["pretrained"] = None
    elif model_type == "pretrained":
        context["pretrained"] = config.get("pretrainedModel", {})
        context["layers"] = []

    # Select templates by domain
    if context["fileMetadata"].get("type", "").startswith("image"):
        preprocessing_template = env.get_template("transforms/vision_transforms.j2")
        model_template = env.get_template("models/vision_models.j2")
    else:
        preprocessing_template = env.get_template("transforms/text_models.j2")
        model_template = env.get_template("models/text_models.j2")

    training_template = env.get_template("training_config.j2")
    inference_template = env.get_template("inference.j2")

    rendered = []
    rendered.append(preprocessing_template.render(**context))
    rendered.append(model_template.render(**context))
    rendered.append(training_template.render(**context))
    rendered.append(inference_template.render(**context))

    code = "\n\n".join(rendered)

    with open(output_path, "w") as f:
        f.write(code)
