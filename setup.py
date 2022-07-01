# -*- coding: utf-8 -*-
__all__ = []

from distutils.core import setup

from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = False

common_compiler_directives = {
    "language_level": 3,
    "c_string_type": "str",
    "c_string_encoding": "utf-8",

    "warn.unreachable": True,
    "warn.unused": True,
    "warn.unused_arg": True,
    "warn.unused_result": True
}
