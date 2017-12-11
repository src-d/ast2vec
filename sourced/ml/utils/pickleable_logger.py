import logging


class PickleableLogger:
    """
    Base class which provides the logging features through ``self._log``.
    Can be safely pickled.
    """
    def __init__(self, log_level=logging.DEBUG):
        self._log = logging.getLogger(self._get_log_name())
        self._log.setLevel(log_level)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_log"] = self._log.level
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        log_level = state["_log"]
        self._log = logging.getLogger(self._get_log_name())
        self._log.setLevel(log_level)

    def _get_log_name(self):
        """
        Children must implement this method. It shall return the logger's name.
        """
        raise NotImplementedError
