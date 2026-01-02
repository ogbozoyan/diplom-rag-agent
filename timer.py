from __future__ import annotations

import logging
import time

_log = logging.getLogger(__name__)


class timed:
    def __init__( self, label: str ):
        self.label = label
        self.t0 = 0.0

    def __enter__( self ):
        self.t0 = time.time()
        _log.info("%s: start", self.label)
        return self

    def __exit__( self, exc_type, exc, tb ):
        dt = time.time() - self.t0
        if exc:
            _log.exception("%s: failed in %.3fs", self.label, dt)
        else:
            _log.info("%s: done in %.3fs", self.label, dt)
        return False
