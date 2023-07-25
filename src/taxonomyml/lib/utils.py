"""Utility functions for the app."""

from __future__ import annotations

import datetime
import glob
import logging
import os
import re
from collections.abc import Collection, Iterator
from itertools import product
from typing import List

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


class DateRange(Collection):
    """An iterable date range.

    Attributes:
        start: The earliest date of the range.
        end: The latest date of the range

    Usage:
        date_range = DateRange('2023-02-01', '2023-02-28')
        # or from a number of days in the past
        date_range = DateRange.from_past_days(90)

        # Iterate over the dates
        for date in date_range:
            print(date)

        # Check for presence of a date
        if '2022-04-05' in date_range:
            print('Date is present in range.')

        # Get the number of days (inclusive) in the range
        range_length = len(date_range)
    """

    def __init__(self, start: str | datetime.date, end: str | datetime.date) -> None:
        self._start = self._to_date_object(start)
        self._end = self._to_date_object(end)
        if self._start > self._end:
            raise ValueError(
                "The start date must be before or the same as the end date."
            )

    @classmethod
    def from_past_days(cls, days: int, offset: int = 0) -> DateRange:
        """Creates a DateRange instance for the past `n` days."""
        today = datetime.date.today()
        count_days = datetime.timedelta(days=abs(days))
        offset_days = datetime.timedelta(days=offset)
        return cls((today - count_days + offset_days), (today + offset_days))

    @property
    def start(self) -> datetime.date:
        return self._start

    @start.setter
    def start(self, date: str | datetime.date) -> None:
        self._start = self._to_date_object(date)

    @property
    def end(self) -> datetime.date:
        return self._end

    @end.setter
    def end(self, date: str | datetime.date) -> None:
        self._end = self._to_date_object(date)

    @staticmethod
    def _to_date_object(date: str | datetime.date) -> datetime.date:
        """Converts an ISO date string to a datetime.date object."""
        if isinstance(date, str):
            return datetime.date.fromisoformat(date)
        return date

    def __contains__(self, comparison: str | datetime.date) -> bool:
        return (
            True
            if self.start <= self._to_date_object(comparison) <= self.end
            else False
        )

    def __iter__(self) -> Iterator[datetime.date]:
        for n in range(self.start.toordinal(), self.end.toordinal() + 1):
            yield datetime.date.fromordinal(n)

    def __len__(self) -> int:
        return (self.end.toordinal() - self.start.toordinal()) + 1

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(start='{self.start}', " f"end='{self.end}')>"
