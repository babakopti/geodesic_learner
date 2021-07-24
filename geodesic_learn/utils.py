# Import libs
import logging

# get_logger(): Create a logger object
def get_logger(name, level=logging.INFO):

    logger = logging.getLogger(name)

    logger.handlers = []

    logger.setLevel(level)

    f_hd = logging.StreamHandler()
    log_fmt = logging.Formatter(
        "%(asctime)s - %(name)s %(levelname)-s - %(message)s"
    )

    f_hd.setFormatter(log_fmt)
    logger.addHandler(f_hd)

    return logger
