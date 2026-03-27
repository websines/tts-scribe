from __future__ import annotations

import logging
import sys

from tts_server.config import settings


def setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    fmt = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy libs
    for name in ("httpx", "httpcore", "urllib3", "filelock", "transformers.tokenization_utils"):
        logging.getLogger(name).setLevel(logging.WARNING)
