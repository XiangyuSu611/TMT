import logging


LOG_FORMAT= "[%(asctime)s] [%(levelname)7s] %(message)s (%(name)s:%(lineno)s)"
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_logger_set = False


def init_logger(name):
    global _logger_set
    if _logger_set:
        return logging.getLogger(name)
    _logger_set = True
    LOG_LEVEL = logging.INFO
    formatter = logging.Formatter(LOG_FORMAT, LOG_TIME_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(console)
    logger.setLevel(LOG_LEVEL)
    return logging.getLogger(name)


def disable_logging(name):
    logging.getLogger(name).setLevel(logging.ERROR)
