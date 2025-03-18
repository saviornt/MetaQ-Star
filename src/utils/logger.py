# ./utils/logger.py

import logging
from src.settings.logging_settings import LoggingSettings

# Instantiate logging settings
settings = LoggingSettings()

# Define custom log level for metrics
METRICS_LEVEL = 25
logging.addLevelName(METRICS_LEVEL, "METRICS")

# Add metrics method to Logger class
def metrics(self, msg, *args, **kwargs):
    """
    Log a message with METRICS level.

    Args:
        msg: The message to log
        args: Additional positional arguments
        kwargs: Additional keyword arguments
    """
    if self.isEnabledFor(METRICS_LEVEL):
        self._log(METRICS_LEVEL, msg, args, **kwargs)

# Add the metrics method to the Logger class
logging.Logger.metrics = metrics


def setup_logger(name: str, level: str = None) -> logging.Logger:
    """
    Configures and returns a logger with the specified name.

    Args:
        name (str): The name of the logger. Typically, use `__name__` to get the module's name.
        level (str): Overrides the log level defined in settings. If not provided, defaults to settings.

    Returns:
        logging.Logger: A configured logger instance.
    
    Usage Example:
    --------------
    logger = setup_logger(__name__)

    logger.info("This is an INFO message.")

    logger.debug("This is a DEBUG message.")
    
    logger.warning("This is a WARNING message.")

    logger.exception(f"This is how to log an exception: {e}")

    logger.error("This is an ERROR message.")
    
    logger.critical("This is a CRITICAL message.")
    
    logger.metrics("This is a METRICS message.")
    """

    logger = logging.getLogger(name)

    # If the logger already has handlers, return it to prevent duplicate logs
    if logger.handlers:
        return logger

    # Determine the log level
    log_level = level.upper() if level else settings.level.upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(numeric_level)

    # Create a console handler
    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)

    # Create a formatter based on settings
    formatter = logging.Formatter(
        fmt=settings.format,
        datefmt=settings.datefmt,
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Prevent log messages from propagating to the root logger
    logger.propagate = False

    return logger
