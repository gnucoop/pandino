
import os
import logging

def setup_datachat_runtime_logger() -> logging.Logger:
    """
    Configure runtime logger for DataChat observability on terminal.
    Keeps existing structured/file logging untouched.
    """
    logger = logging.getLogger("datachat.runtime")
    logger.setLevel(getattr(logging, os.getenv("DATACHAT_LOG_LEVEL", "INFO").upper(), logging.INFO))
    logger.propagate = False

    if not any(getattr(h, "_datachat_runtime", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler._datachat_runtime = True  # type: ignore[attr-defined]
        handler.setLevel(logger.level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)

    return logger
