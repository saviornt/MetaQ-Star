# test_logger.py

import sys
from pathlib import Path

try:
    from src.utils.logger import setup_logger
except ImportError:
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.logger import setup_logger


def main():
    """
    Test the logging setup by generating log messages at different levels.
    """
    logger = setup_logger(__name__)

    logger.info("This is an INFO message.")
    logger.debug("This is a DEBUG message.")
    logger.warning("This is a WARNING message.")
    
    # Proper way to use logger.exception - inside an except block
    try:
        1/0  # Deliberately cause a ZeroDivisionError
    except Exception as e:
        logger.exception(f"This is how to log an exception: {e}")
    
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")
    logger.metrics("This is a METRICS message.")
    

if __name__ == "__main__":
    main()
