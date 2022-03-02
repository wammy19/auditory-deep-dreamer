from __future__ import annotations

from time import perf_counter
from typing import Callable, Optional


class Timer:
    """
    Timer class used for benchmarking blocks of code.

    Resource: https://www.geeksforgeeks.org/python-how-to-time-the-program/
    """


    # =================================================================================================================
    # ---------------------------------------------- Class Constructors -----------------------------------------------
    # =================================================================================================================

    def __init__(self, func: Callable = perf_counter):
        """
        :param func: A callable function that tracks time. Anything from the 'time' module will do.
        """

        self._elapsed: float = 0.0
        self._func = func
        self._start: Optional[Callable] = None


    # =================================================================================================================
    # ---------------------------------------------- Public Functions -------------------------------------------------
    # =================================================================================================================

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


    # =================================================================================================================
    # ---------------------------------------------- Getter Functions -------------------------------------------------
    # =================================================================================================================

    @property
    def get_elapsed_time(self) -> float:
        """
        :return:

        Returns total time taken after .stop() has been called.
        Calling this method restarts the timer.
        """

        self.stop()
        self._start = None

        return self._elapsed


    # =================================================================================================================
    # ---------------------------------------------- Dunder Functions -------------------------------------------------
    # =================================================================================================================

    def __enter__(self) -> Timer:
        """
        :return:
        """

        self.start()

        return self


    def __exit__(self, *args) -> None:
        """
        :param args:
        :return:
        """

        self.stop()


    # =================================================================================================================
    # ---------------------------------------------- Decorator Functions ----------------------------------------------
    # =================================================================================================================

    @staticmethod
    def time_func(func: Callable) -> Callable:
        """
        :param func: Any callable function for timing.
        :return:

        This function is intended to be used as a decorator function for timing functions.
        """


        def wrapper(*args, **kwargs) -> None:
            timer = Timer()
            timer.start()

            func(*args, **kwargs)

            print(f'Time took for {func.__name__}: {round(timer.get_elapsed_time, 5)} seconds.')


        return wrapper
