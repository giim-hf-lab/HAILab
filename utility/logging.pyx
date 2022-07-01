# -*- coding: utf-8 -*-
__all__ = ["LoggerIO"]

from io import TextIOBase
from logging import getLogger, Logger


class LoggerIO(TextIOBase):
    __slots__ = "_level", "_logger"

    encoding: str = "utf-8"

    _level: int
    _logger: Logger

    def __init__(self, str name, size_t level) -> None:
        TextIOBase.__init__(self)

        self._level = level
        self._logger = getLogger(name)

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

    def write(self, str s) -> int:
        cdef str split
        for split in s.splitlines():
            split = split.rstrip()
            if split:
                self._logger.log(self._level, split)
        return len(s)
