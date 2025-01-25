import logging
import os
from typing import Optional

import colorama

# Initialize colorama for Windows support
colorama.init()


class ColoredLogger:
    """
    A custom logger class that provides colored console output and file logging.

    Features:
    - Colored console output for different log levels
    - Optional file logging with customizable file path
    - Timestamp formatting
    - Multiple log level support
    """

    # Color scheme for different log levels
    COLORS = {
        'DEBUG': colorama.Fore.BLUE,
        'INFO': colorama.Fore.WHITE,
        'WARNING': colorama.Fore.YELLOW,
        'ERROR': colorama.Fore.RED,
        'CRITICAL': colorama.Fore.RED + colorama.Style.BRIGHT
    }

    class ColoredFormatter(logging.Formatter):
        """Custom formatter that adds colors to log messages for console output"""

        def __init__(self, colors, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.colors = colors

        def format(self, record):
            # Add color to the level name and message
            if record.levelname in self.colors:
                color = self.colors[record.levelname]
                record.levelname = f"{color}{record.levelname}{colorama.Style.RESET_ALL}"
                record.msg = f"{color}{record.msg}{colorama.Style.RESET_ALL}"
            return super().format(record)

    def __init__(self,
                 name: str,
                 level: str = 'INFO',
                 log_file: Optional[str] = None,
                 log_format: str = '%(asctime)s - %(levelname)s - %(message)s',
                 date_format: str = '%Y-%m-%d %H:%M:%S'):
        """
        Initialize the colored logger.

        Args:
            name: Logger name (typically __name__)
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for logging. If None, only console logging is used
            log_format: Format string for log messages
            date_format: Format string for timestamps
        """
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Console handler with colored output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            self.ColoredFormatter(self.COLORS, log_format, date_format))
        self.logger.addHandler(console_handler)

        # File handler (if log_file specified)
        if log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(log_format, date_format))
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)


# Example usage:
if __name__ == '__main__':
    # Create a logger with both console and file output
    logger = ColoredLogger(name=__name__,
                           level='DEBUG',
                           log_file='logs/app.log')

    # Test all log levels
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
