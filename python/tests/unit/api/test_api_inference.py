import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from python.src.api.inference import (
    BatchInferenceHandler,
    PredictionRequest,
    app,
    get_model,
)

# Disable telemetry for tests
os.environ["NGLAB_ENABLE_TELEMETRY"] = "false"


@pytest.fixture
def client():
    # Mock Redis to return None for get() so it doesn't "hit" the cache
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.close = AsyncMock()

    with patch("python.src.api.inference._REDIS", mock_redis):
        # Also prevent lifespan from overwriting it with a real connection
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            with TestClient(app) as c:
                yield c


def test_prediction_request_validation():
    # Valid
    req = PredictionRequest(observations=[[1.0, 2.0], [3.0, 4.0]])
    assert len(req.observations) == 2

    # Empty - Pydantic 2 uses ValidationError and 'too_short' error
    with pytest.raises(ValidationError):
        PredictionRequest(observations=[])

    # Inconsistent length
    with pytest.raises(ValidationError):
        PredictionRequest(observations=[[1.0], [1.0, 2.0]])


def test_batch_inference_handler_init():
    handler = BatchInferenceHandler(lambda: MagicMock())
    assert handler.queue is None
    assert not handler._shutdown


@pytest.mark.anyio
async def test_batch_inference_handler_start_stop():
    handler = BatchInferenceHandler(lambda: MagicMock())
    await handler.start()
    assert handler.queue is not None
    assert handler._worker_task is not None
    await handler.stop()
    assert handler._shutdown


@pytest.mark.anyio
async def test_batch_inference_handler_process_batch():
    mock_model = MagicMock()
    # Mock model output: expect [2, 2] output for [2, 10] input
    mock_model.side_effect = lambda x: torch.ones(x.shape[0], 2)
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(1))])

    handler = BatchInferenceHandler(lambda: mock_model)
    await handler.start()

    req = PredictionRequest(observations=[[0.1] * 10])
    future = asyncio.get_running_loop().create_future()
    batch = [(req, future)]

    await handler._process_batch(batch)

    assert future.done()
    res = await future
    assert len(res) == 1
    assert len(res[0]) == 2
    await handler.stop()


def test_get_model_singleton():
    with patch("python.src.api.inference.load_model") as mock_load:
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_load.return_value = (mock_model, {})

        # Reset global state for test
        from python.src.api import inference

        inference._MODEL = None

        m1 = get_model()
        m2 = get_model()

        assert m1 == m2
        assert mock_load.call_count == 1


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "gpu_available" in data


@patch("python.src.api.inference.batch_handler.predict", new_callable=AsyncMock)
def test_predict_endpoint(mock_predict, client):
    mock_predict.return_value = [[0.5, 0.6]]

    payload = {"observations": [[0.1, 0.2, 0.3]], "temperature": 1.5}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == [[0.5, 0.6]]
    assert data["cached"] is False


def test_predict_endpoint_cached(client):
    # Here we want a hit, so we need to configure the mock_redis
    # But 'client' fixture already provides a mock_redis via patching.
    # We can patch it again locally or just use a new one.

    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps([[0.9, 0.1]])

    with patch("python.src.api.inference._REDIS", mock_redis):
        with patch("python.src.api.inference.batch_handler.predict") as mock_predict:
            payload = {"observations": [[0.1, 0.2, 0.3]]}
            response = client.post("/predict", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert data["predictions"] == [[0.9, 0.1]]
            assert data["cached"] is True
            mock_redis.get.assert_called()
            mock_predict.assert_not_called()
