"""
pytest configuration and fixtures for test suite.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (requiring live server)"
    )


@pytest.fixture
def url():
    """
    URL fixture for tests.
    This is used by integration tests that are typically run manually.
    """
    pytest.skip("Integration tests require manual execution with a running server")
