from astrodata.preml.processors.MissingImputator import MissingImputator
from astrodata.preml.processors.Ohe import OHE
from astrodata.preml.processors.Standardizer import Standardizer
from astrodata.preml.processors.TrainTestSplitter import TrainTestSplitter


def instantiate_processors(
    config: dict, ignore_unknown: bool = True, defaults: dict = None
) -> dict:
    """
    Given a config dict, returns a dict mapping processor names to instances.
    Validates processor names, catches instantiation errors, and allows for defaults.

    Args:
        config (dict): The 'preml' section of the configuration.
        ignore_unknown (bool): If True, unknown processors are ignored. If False, raises error.
        defaults (dict): Optional default parameters for processors.

    Returns:
        dict: Dictionary mapping processor names to their instances.

    Raises:
        ValueError: If unknown processor is found and ignore_unknown is False.
        RuntimeError: If processor instantiation fails.
    """
    if config == {}:
        return {}
    processor_classes = {
        "TrainTestSplitter": TrainTestSplitter,
        "OHE": OHE,
        "MissingImputator": MissingImputator,
        "Standardizer": Standardizer,
    }
    instances = {}
    defaults = defaults or {}
    for name, params in config.items():
        cls = processor_classes.get(name)
        if not cls:
            if ignore_unknown:
                continue
            else:
                raise ValueError(
                    f"Unknown processor: '{name}'. Available: {list(processor_classes.keys())}"
                )
        # Merge defaults if provided
        merged_params = {**defaults.get(name, {}), **params}
        try:
            instances[name] = cls(**merged_params)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate processor '{name}': {e}")
    return instances
