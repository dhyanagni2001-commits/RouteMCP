from pathlib import Path

from routemcp.domain import ModelConfig
from routemcp.registry import ModelRegistry


def configured_model(model_id: str) -> ModelConfig:
    return ModelConfig(id=model_id, provider="mock", model=model_id)


def test_api_models_persist_across_registry_restarts(tmp_path: Path) -> None:
    database = tmp_path / "registry.db"
    first = ModelRegistry(storage_path=database)
    first.register(configured_model("persistent"), persist=True)
    first.close()

    second = ModelRegistry(storage_path=database)
    assert second.get("persistent").model == "persistent"
    second.remove("persistent", persist=True)
    second.close()

    third = ModelRegistry([configured_model("persistent")], storage_path=database)
    assert third.list() == []
    third.close()


def test_circuit_breaker_opens_and_resets() -> None:
    registry = ModelRegistry([configured_model("unstable")], failure_threshold=2)
    registry.record_failure("unstable", "timeout")
    assert registry.health("unstable")["circuit_open"] is not True
    registry.record_failure("unstable", "timeout")
    assert registry.health("unstable")["circuit_open"] is True
    registry.record_success("unstable")
    assert registry.health("unstable")["circuit_open"] is False
