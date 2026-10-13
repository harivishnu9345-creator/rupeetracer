import pytest

from data import generator


@pytest.fixture(scope="session")
def small():
    """Small deterministic dataset (1,000 accounts) shared by all tests."""
    return generator.generate(n_accounts=1000, seed=42)
