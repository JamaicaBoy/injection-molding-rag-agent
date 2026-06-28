from pathlib import Path

import yaml


def test_default_models_are_local() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "app_config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["llm"]["provider"] == "ollama"
    assert config["llm"]["model"]
    assert config["embedding"]["provider"] == "sentence-transformers"
    assert config["embedding"]["model"]
