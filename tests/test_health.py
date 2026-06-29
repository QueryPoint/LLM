from fastapi.testclient import TestClient

from assistant_service.main import app


def test_health_live_returns_200() -> None:
    client = TestClient(app)

    response = client.get("/health/live")

    assert response.status_code == 200


def test_health_live_returns_expected_payload() -> None:
    client = TestClient(app)

    response = client.get("/health/live")

    assert response.json() == {
        "status": "ok",
        "service": "assistant-service",
    }


def test_health_ready_returns_503_when_worker_is_not_ready() -> None:
    client = TestClient(app)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service": "assistant-service",
        "rabbitmq": "disconnected",
    }
