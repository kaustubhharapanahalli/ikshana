import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not os.path.exists("logs"):
    os.mkdir("logs")

# Create handlers for logging
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)

logger_file_handler = logging.FileHandler(os.path.join("logs", "file.log"))
logger_file_handler.setLevel(logging.INFO)


# Create a Formatter for formatting the log messages
logger_stream_formatter = logging.Formatter(
    "%(name)s - [%(levelname)s]: %(message)s"
)
logger_file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Add the Formatter to the Handler
logger_stream_handler.setFormatter(logger_stream_formatter)
logger_file_handler.setFormatter(logger_file_formatter)

# Add the Handler to the Logger
logger.addHandler(logger_stream_handler)
logger.addHandler(logger_file_handler)

logger.info("Logging module setup and ready.")
