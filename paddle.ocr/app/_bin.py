# -*- coding: utf-8 -*-
__all__ = ["main"]

import asyncio
import logging.config
import os.path
import sys
from argparse import ArgumentParser, Namespace
from io import TextIOBase
from typing import Any, Dict

from hypercorn.asyncio import serve
from hypercorn.config import Config

from ._server import server


class LoggerIO(TextIOBase):
    __slots__ = "_level", "_logger"

    encoding: str = "utf-8"

    _level: str
    _logger: logging.Logger

    def __init__(self, name: str, level: str) -> None:
        self._level = level
        self._logger = logging.getLogger(name)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return 0

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        self._logger.log(logging.getLevelName(self._level), s)
        return len(s)


_LOG_LEVELS = frozenset({
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL"
})


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("-a", "--access-log-level", default=None, choices=_LOG_LEVELS, dest="access_log_level")
    parser.add_argument("-B", "--backlog", default=10, type=int, dest="backlog")
    parser.add_argument("-l", "--log-level", default="INFO", choices=_LOG_LEVELS, dest="log_level")
    parser.add_argument("-L", "--log-prefix", required=True, dest="log_prefix")
    parser.add_argument("-n", "--name", default="server", dest="name")
    parser.add_argument("-p", "--port", default=80, type=int, dest="port")

    parser.add_argument("--log-stderr", default="ERROR", choices=_LOG_LEVELS, dest="log_stderr")
    parser.add_argument("--log-stdout", default="INFO", choices=_LOG_LEVELS, dest="log_stdout")

    args: Namespace = parser.parse_args()

    logconfig: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "incremental": False,
        "formatters": {
            "default": {
                "class": "logging.Formatter",
                "format": "[%(asctime)s (%(levelname)s)] %(name)s >> %(message)s"
            },
            "noname": {
                "class": "logging.Formatter",
                "format": "[%(asctime)s (%(levelname)s)] >> %(message)s"
            }
        },
        "root": {
            "level": args.log_level,
            "handlers": ["root"]
        },
        "handlers": {
            "root": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": os.path.join(args.log_prefix, f"{args.name}.log"),
                "when": "W0",
                "backupCount": 1,
                "encoding": "utf-8",
                "formatter": "default"
            }
        },
        "loggers": {
            "ppocr": {
                "level": args.log_level,
                "handlers": [],
                "propagate": True
            }
        }
    }
    handlers: Dict[str, Dict[str, Any]] = logconfig["handlers"]
    loggers: Dict[str, Dict[str, Any]] = logconfig["loggers"]
    for logger_name in (
        f"hypercorn.{hypercorn_type}"
        for hypercorn_type in ("access", "error")
    ):
        handlers[logger_name] = {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": os.path.join(args.log_prefix, f"{logger_name}.{args.name}.log"),
            "when": "W0",
            "backupCount": 1,
            "encoding": "utf-8",
            "formatter": "noname"
        }
        loggers[logger_name] = {
            "level": args.log_level,
            "handlers": [logger_name],
            "propagate": False
        }
    loggers["hypercorn.access"]["level"] = args.access_log_level or args.log_level
    logging.config.dictConfig(logconfig)

    sys.stdout = LoggerIO("sys.stdout", args.log_stdout)
    sys.stderr = LoggerIO("sys.stderr", args.log_stderr)

    config = Config()
    config.accesslog = logging.getLogger("hypercorn.access")
    config.errorlog = logging.getLogger("hypercorn.error")
    config.backlog = args.backlog
    config.bind = f"0.0.0.0:{args.port}"

    asyncio.run(serve(server, config))

    return 0
