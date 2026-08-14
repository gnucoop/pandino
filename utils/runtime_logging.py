
import os
import logging

def setup_datachat_runtime_logger() -> logging.Logger:
    """
    Configure runtime logger for DataChat observability on terminal.
    Keeps existing structured/file logging untouched.
    """
    # Function-local: utils.logging_config imports setup_datachat_runtime_logger
    # at module level, before its own ContextDefaultsFilter class is defined,
    # so a top-level import here would be circular. By call time (from
    # bootstrap_logging(), at the end of that module) logging_config is fully
    # loaded, so the deferred import is safe - same pattern already used by
    # logging_config's own function-local `from flask import g`.
    from utils.logging_config import ContextDefaultsFilter  # noqa: PLC0415

    logger = logging.getLogger("datachat.runtime")
    logger.setLevel(getattr(logging, os.getenv("DATACHAT_LOG_LEVEL", "INFO").upper(), logging.INFO))
    logger.propagate = False

    if not any(getattr(h, "_datachat_runtime", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler._datachat_runtime = True  # type: ignore[attr-defined]
        handler.setLevel(logger.level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s "
                "request_id=%(request_id)s app_id=%(app_id)s %(message)s"
            )
        )
        handler.addFilter(ContextDefaultsFilter())
        logger.addHandler(handler)

    return logger
