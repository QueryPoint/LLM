import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("RUN_INTEGRATION_TESTS", "").strip().lower() == "true":
        return

    skip_integration = pytest.mark.skip(
        reason="integration tests require RUN_INTEGRATION_TESTS=true"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
