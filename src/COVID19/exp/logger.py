import logging

########## log ##########
def make_logger(name, log_path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # root logger로 전파 방지

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fmt = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger