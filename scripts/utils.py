import logging


def get_logger(level, name):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        level=level
    )
    return logging.getLogger(name)
