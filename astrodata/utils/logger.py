import logging
import sys


def setup_logger(name: str = "astrodata", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger with the specified name and level.
    Logs to stdout with a simple format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
