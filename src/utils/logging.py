import logging
import os


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


class LevelsFilter(logging.Filter):
    def __init__(self, levels):
        self.levels = [getattr(logging, level) for level in levels]

    def filter(self, record):
        return record.levelno in self.levels


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        import tqdm

        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def start_disable_output(logfile):
    # Open the logfile for append
    with open(logfile, "a") as log_file:
        # Save the original stdout file descriptor
        original_stdout = os.dup(1)

        # Redirect stdout to the log file
        os.dup2(log_file.fileno(), 1)

        # Return the original stdout file descriptor
        return original_stdout


def stop_disable_output(original_stdout):
    # Restore the original stdout file descriptor
    os.dup2(original_stdout, 1)
