"""Utility functions for the app."""

from __future__ import annotations

import re
import os
import glob
from typing import List
from itertools import product
import logging
from loguru import logger


class InterceptHandler(logging.Handler):
    """Special handler to send builtin logging messages to loguru."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """Sets up logging for the application."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    logger.disable("google.auth.transport.requests")


def create_tuples(l1: List[str], l2: List[str]) -> List[List[str]]:
    """Create tuples from two lists."""
    return [[item1, item2] for item1, item2 in product(l1, l2) if item1 != item2]


def process_result(result: str) -> dict:
    """Takes a string in the following format:
    %%formula_description_html%%: <div class="formula">...</div>
    %%example_table_html%%: <div class="table">...</div>
    Converts to a dict in the following format:
    {'formula_description_html': '<div class="formula">...</div>', 'example_table_html': '<div class="table">...</div>'}
    """
    result_dict = {}
    variable_key = None

    # Iterate line by line over the prompt
    for line in re.split(r"\n+", result):
        variable = re.findall(r"%%([^%]+)%%:", line)
        if variable:
            variable_key = variable[0]
            result_dict[variable_key] = [line.split(":", 1)[1]]
        elif variable_key:
            result_dict[variable_key].append(line)

    # Join the lines together
    for key, value in result_dict.items():
        result_dict[key] = "\n\n".join([v.strip() for v in value]).strip()

    return result_dict


def get_credentials_path(credentials_file: str) -> str:
    """Get the path to the credentials file."""

    matches = glob.glob(f"**/{credentials_file}", recursive=True)
    matches = matches + glob.glob(f"../**/{credentials_file}", recursive=True)

    if len(matches) > 0:
        return os.path.abspath(matches[0])
    else:
        raise FileNotFoundError("Credentials file not found.")
