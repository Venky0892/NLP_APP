import logging

logger = logging.getLogger()
def log(message, level=None):
    if level is None:
        logger.warning(message)

    if level == "debug":
        logger.debug(message)

    if level == "info":
        logger.info(message)

    if level == "warning":
        logger.warning(message)

    if level == "error":
        logger.error(message)

    if level == "critical":
        logger.critical(message)