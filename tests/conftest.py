"""Shared pytest fixtures for the ACRF test suite."""

import pytest
from fastapi.testclient import TestClient
from main import app

from app.experience import DEFAULT_EXPERIENCE_REPOSITORY
from app.metrics import DEFAULT_METRICS_REPOSITORY


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client for exercising the API layer in tests."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_default_experience_repository() -> None:
    """Clear the process-wide `DEFAULT_EXPERIENCE_REPOSITORY` before every test.

    `evaluation_node` always records into this shared singleton (see
    `app/experience/repository.py`), so without this fixture, tests using
    the same `session_id`/`task_id`/`iteration_count` fixture values would
    collide on `experience_id` uniqueness across otherwise-unrelated
    tests. Runs automatically for every test in the suite.
    """
    DEFAULT_EXPERIENCE_REPOSITORY.clear()


@pytest.fixture(autouse=True)
def _clear_default_metrics_repository() -> None:
    """Clear the process-wide `DEFAULT_METRICS_REPOSITORY` before every test.

    `evaluation_node` always records into this shared singleton (see
    `app/metrics/repository.py`). Unlike `DEFAULT_EXPERIENCE_REPOSITORY`,
    it does not reject duplicates, so without this fixture, leftover
    records from earlier tests would silently inflate `count()`/`list()`/
    `summary()` results in unrelated tests. Runs automatically for every
    test in the suite.
    """
    DEFAULT_METRICS_REPOSITORY.clear()
