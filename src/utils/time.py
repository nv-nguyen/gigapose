# credit: https://github.com/shanice-l/gdrnpp_bop2022/blob/f3ca18632f4b68c15ab3306119c364a0497282a7/lib/utils/time_utils.py#L51
import time
from datetime import timedelta
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_time_delta(sec):
    """Humanize timedelta given in seconds, modified from maskrcnn-
    benchmark."""
    if sec < 0:
        logger.warning("get_time_delta() obtains negative seconds!")
        return "{:.3g} seconds".format(sec)
    delta_time_str = str(timedelta(seconds=sec))
    return delta_time_str


class Timer(object):
    # modified from maskrcnn-benchmark
    def __init__(self):
        self.reset()

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=False):
        self.add(time.time() - self.start_time)

        if average:
            return self.average_time
        else:
            return self.diff

    def add(self, time_diff):
        self.diff = time_diff
        self.total_time += self.diff
        self.calls += 1

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0

    def avg_time_str(self):
        time_str = get_time_delta(self.average_time)
        return time_str
