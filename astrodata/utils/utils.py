from pathlib import Path

import yaml


def read_config(config_path: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.
    Ensures that a 'project_path' key exists and resolves it to an absolute path.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The contents of the YAML file as a dictionary, with 'project_path' absolute.

    Raises:
        KeyError: If 'project_path' is missing in the config.
    """

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    if "project_path" not in config:
        raise KeyError("Missing required 'project_path' key in config file.")
    config["project_path"] = Path(config["project_path"]).resolve()
    return config


def get_output_path(project_path: Path, subfolder: str, filename: str = None) -> Path:
    output_dir = project_path / "astrodata_files" / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    if not filename:
        return output_dir
    return output_dir / filename
