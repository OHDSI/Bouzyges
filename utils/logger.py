import logging
import sys


## Logging
LOGGER = logging.getLogger("Bouzyges")
logging.basicConfig(level=logging.INFO)
LOGGER.info("Logging started")
# Default handler and formatter
LOGGER.handlers.clear()
_stdout_handler = logging.StreamHandler(sys.stdout)
FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
)
_stdout_handler.setFormatter(FORMATTER)
LOGGER.addHandler(_stdout_handler)
LOGGER.info("Logging configured")
