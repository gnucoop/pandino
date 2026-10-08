import os
import logging


def setup_runtime_logger(name: str, level_env_var: str = "DATACHAT_LOG_LEVEL") -> logging.Logger:
    """
    Configure a runtime logger for agent observability on terminal.
    Keeps existing structured/file logging untouched.

    :param name: logger name, e.g. "datachat.runtime" or "interviewer.runtime".
    :param level_env_var: env var holding the log level (default INFO).
    :return: the configured logger; calling again never adds a second handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, os.getenv(level_env_var, "INFO").upper(), logging.INFO))
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


def setup_datachat_runtime_logger() -> logging.Logger:
    """
    Configure runtime logger for DataChat observability on terminal.
    Keeps existing structured/file logging untouched.
    """
    return setup_runtime_logger("datachat.runtime")


def setup_interviewer_runtime_logger() -> logging.Logger:
    """Configure runtime logger for the analysis interviewer."""
    return setup_runtime_logger("interviewer.runtime", level_env_var="INTERVIEWER_LOG_LEVEL")
