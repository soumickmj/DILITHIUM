#!/usr/bin/env python
"""

Purpose :

"""

import logging


class Logger:
    def __init__(self, model_name, logger_path):
        self.logger = logging.getLogger(model_name)
        log_handler = logging.FileHandler(logger_path)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        log_handler.setFormatter(formatter)
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logging.DEBUG)

    def get_logger(self):
        return self.logger
