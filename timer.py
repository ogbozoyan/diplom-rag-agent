from __future__ import annotations

import time

from logger_config import logger


class timed:
    def __init__( self, label: str ):
        self.label = label
        self.t0 = 0.0

    def __enter__( self ):
        self.t0 = time.time()
        logger.info("%s: start", self.label)
        return self

    def __exit__( self, exc_type, exc, tb ):
        dt = time.time() - self.t0
        if exc:
            logger.exception("%s: failed in %.3fs", self.label, dt)
        else:
            logger.info("%s: done in %.3fs", self.label, dt)
        return False
