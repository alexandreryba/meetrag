from __future__ import annotations

import json

from fastapi.testclient import TestClient
from pydantic import SecretStr

from polyrag.api.app import create_app
from polyrag.container import Container


def _client(container: Container) -> TestClient:
    return TestClient(create_app(container))


def test_health_and_readiness(container: Container) -> None:
    with _client(container) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["chunks"] > 0
        assert client.get("/").headers["content-type"].startswith("text/html")


def test_ask_returns_a_cited_answer(container: Container) -> None:
    with _client(container) as client:
        response = client.post("/v1/ask", json={"question": "What is the hotel cap in Paris?"})
        assert response.status_code == 200
        body = response.json()
        assert (
            body["citations"] and body["usage"]["llm_calls"] == 0
        )  # scripted LLM records no usage
        assert "x-request-id" in response.headers


def test_ask_validates_input(container: Container) -> None:
    with _client(container) as client:
        assert client.post("/v1/ask", json={"question": ""}).status_code == 422


def test_stream_uses_server_sent_events(container: Container) -> None:
    with _client(container) as client:
        response = client.post("/v1/ask/stream", json={"question": "What is the wallbox price?"})
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1]["type"] == "final"


def test_api_key_is_enforced_when_configured(container: Container) -> None:
    container.settings = container.settings.model_copy(update={"api_key": SecretStr("s3cret")})
    with _client(container) as client:
        assert client.get("/v1/sources").status_code == 401
        response = client.get("/v1/sources", headers={"X-API-Key": "s3cret"})
        assert response.status_code == 200
        assert {s["name"] for s in response.json()} == {
            "meetings",
            "policies",
            "products",
            "support",
            "analytics",
        }


def test_missing_llm_credentials_return_503(container: Container) -> None:
    from polyrag.domain.ports import LLMConfigurationError

    class Unconfigured:
        async def ask(self, question: str) -> None:
            raise LLMConfigurationError("no Claude credentials found")

    container.__dict__["assistant"] = Unconfigured()
    with _client(container) as client:
        response = client.post("/v1/ask", json={"question": "What is the hotel cap?"})
        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]
