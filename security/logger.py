# -*- coding: utf8 -*-

__all__  = ["ExecuteLogger"]

import os
import logging

class ExecuteLogger(object):
    def __init__(self, dir=None):
        if dir is None:
            dir = os.path.split(os.path.realpath(__file__))[0] + "\\log\\"
        self.dir = dir
        if os.path.exists(self.dir) is False:
            os.mkdir(self.dir)
        self.logger_dicr = {}

    def get_logger(self, name="info"):
        self.name = name
        logger_path = self.dir + name + ".log"
        _logger = self.logger_dicr.get(logger_path, None)
        if _logger is None:
            _logger = logging.getLogger(self.name)
            _handler = logging.FileHandler(filename=logger_path, mode="a")
            _formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(lineno)s][%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            _handler.setFormatter(_formatter)
            _logger.addHandler(_handler)
            _logger.setLevel(logging.DEBUG)
            self.logger_dicr[logger_path] = _logger
        return _logger
