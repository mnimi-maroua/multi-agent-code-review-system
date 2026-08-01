"""
Step 6: Shared logging configuration.

Every entry-point script (fetch_prs.py, baseline_multi_model.py,
multi_agent_baseline.py, graph.py, evaluate.py) was only logging to the
console -- once the terminal closed, that history was gone. This adds a
persistent log file per run, in addition to the console output, so runs
can be inspected later (e.g. to check exactly which PRs hit rate limits,
or which agent got truncated, without having to re-run anything).


"""

import logging
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")


def setup_logging(script_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging with a console handler and a per-run file handler."""
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{script_name}_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Avoid duplicate handlers if setup_logging is somehow called twice
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(script_name)
    logger.info("Logging to %s", log_file)
    return logger