from __future__ import annotations
import logging as logger
from time import perf_counter
from typing import Callable, Optional


class Timer:
    """
    Timer class used for benchmarking blocks of code.

    Resource: https://www.geeksforgeeks.org/python-how-to-time-the-program/
    """

    def __init__(self, func: Callable = perf_counter):
        """
        :param: func: A callable function that tracks time. Anything from the 'time' module will do.
        """

        self._elapsed: float = 0.0
        self._func = func
        self._start: Optional[Callable] = None


    def start(self) -> None:
        """
        :return:

        Start timer.
        """

        if self._start is not None:
            raise RuntimeError('Already started')

        self._start = self._func()


    def stop(self) -> None:
        """
        :return:

        Stop timer.
        """

        if self._start is None:
            raise RuntimeError('Not started')

        end = self._func()

        self._elapsed += end - self._start
        self._start = None


    def reset(self) -> None:
        """
        :return:

        Resets the timer.
        """

        self._elapsed = 0.0


    def is_running(self) -> bool:
        """
        :return:

        Returns true if the timer is still running.
        """

        return self._start is not None


    @property
    def elapsed(self) -> float:
        """
        :return:

        Returns total time taken after .stop() has been called.
        """

        if self.is_running():
            logger.warning('Timer is still running. Call the .stop() method before getting the elapsed time.')

        return self._elapsed


    def __enter__(self) -> Timer:
        self.start()

        return self


    def __exit__(self, *args) -> None:
        self.stop()
