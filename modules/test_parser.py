import subprocess
import tempfile
import json
import os
from pathlib import Path

base_dir = Path(__file__).parent

config_variants = [
    {
        "name": "pretrained_image_classification",
        "config": {
            "mainTask": "classification",
            "subTask": "image",
            "modelType": "pretrained"
        }
    },
    {
        "name": "custom_audio_classification",
        "config": {
            "mainTask": "classification",
            "subTask": "audio",
            "modelType": "custom"
        }
    },
    {
        "name": "pretrained_text_generation",
        "config": {
            "mainTask": "generation",
            "subTask": "text",
            "modelType": "pretrained"
        }
    },
    # Add more variations for testing
]

def run_parser_test(variant):
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(json.dumps(variant["config"]))

        result = subprocess.run(
            ["python", "parser.py", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=base_dir
        )

        assert result.returncode == 0, f"{variant['name']} failed:\n{result.stderr}"
        try:
            output = json.loads(result.stdout.strip())
        except Exception:
            raise AssertionError(f"{variant['name']} produced invalid JSON output:\n{result.stdout}")

        out_path = Path(output["generated_path"])
        assert out_path.exists(), f"{variant['name']} did not create output file"

        content = out_path.read_text()
        assert "import" in content or "def" in content, f"{variant['name']} generated empty or invalid content"

        print(f"{variant['name']}: OK")

if __name__ == "__main__":
    for variant in config_variants:
        run_parser_test(variant)
