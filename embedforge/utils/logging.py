"""Tiny logger."""
import logging
import sys

_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name="embedforge", level="INFO"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger
