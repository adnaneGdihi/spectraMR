"""LoggingService must not touch process-global logging state.

It used to replace ``logging.getLogger`` and add a root filter and a WARNING default
handler per instance; ``log_policy`` now owns the level, so constructing or closing a
service leaves the stdlib untouched.
"""

from __future__ import annotations

import logging as std_logging

from spectramr.infrastructure.services.logging_service import LoggingService


def test_construction_leaves_the_stdlib_untouched():
    getlogger, filters = std_logging.getLogger, list(std_logging.root.filters)
    handlers, level = list(std_logging.root.handlers), std_logging.root.level
    svc = LoggingService()
    try:
        assert std_logging.getLogger is getlogger
        assert std_logging.getLogger.__module__ == "logging"
        assert list(std_logging.root.filters) == filters
        assert list(std_logging.root.handlers) == handlers
        assert std_logging.root.level == level
    finally:
        svc.close()


def test_a_new_logger_inherits_the_root_level():
    svc = LoggingService()
    try:
        assert std_logging.getLogger("spectramr.fresh_probe").level == std_logging.NOTSET
    finally:
        svc.close()


def test_close_detaches_its_file_handler(tmp_path):
    svc = LoggingService("NoGlobalState")
    svc.setup(str(tmp_path), "info")
    assert any(isinstance(h, std_logging.FileHandler) for h in svc._logger.handlers)
    svc.close()
    assert not any(isinstance(h, std_logging.FileHandler) for h in svc._logger.handlers)


def test_unconfigured_factory_service_has_no_file_sink_and_no_global_effect():
    from spectramr.infrastructure.services.logging_service import LoggingServiceFactory

    level, handlers = std_logging.root.level, list(std_logging.root.handlers)
    svc = LoggingServiceFactory.create_unconfigured("hpo")
    try:
        assert not any(isinstance(h, std_logging.FileHandler) for h in svc._logger.handlers)
        assert std_logging.root.level == level and list(std_logging.root.handlers) == handlers
    finally:
        svc.close()
