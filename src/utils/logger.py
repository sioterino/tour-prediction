import logging
from rich.logging import RichHandler


class RichColoredFormatter(logging.Formatter):
    STYLES = {
        logging.DEBUG: "cyan",
        logging.INFO: "white",
        logging.WARNING: "yellow",
        logging.ERROR: "bold red",
        logging.CRITICAL: "bold white on red",
    }

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        style = self.STYLES.get(record.levelno, "white")
        return f"[{style}]{message}[/{style}]"


def setup_logger(name: str = "app", level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = RichHandler(
        show_time=False,
        show_level=False,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )

    formatter = RichColoredFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    logger = setup_logger()

    file = "example.txt"

    logger.debug("Debug message")
    logger.info("Loading file: %s", file)
    logger.info(f"Loading file with f-string: {file}")
    logger.warning("Disk space is getting low")
    logger.error("Error happened here")
    logger.critical("System is about to explode")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Caught exception")