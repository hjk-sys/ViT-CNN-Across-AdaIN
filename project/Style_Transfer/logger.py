import logging

from colorlog import ColoredFormatter

log_level = logging.INFO
log_format = "  %(log_color)s%(levelname)-8s%(reset)s| %(name)s | %(log_color)s%(message)s%(reset)s | (%(filename)s:%(lineno)d)"
logging.root.setLevel(log_level)
formatter = ColoredFormatter(log_format)
stream = logging.StreamHandler()
stream.setLevel(log_level)
stream.setFormatter(formatter)
log = logging.getLogger("AdaIN 🎨")
log.setLevel(log_level)
log.addHandler(stream)
