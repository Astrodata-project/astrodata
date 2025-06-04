import yaml


def read_config(config_path: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
