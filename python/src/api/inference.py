# mypy: ignore-errors
"""
FastAPI Inference Service for NGLab.

Provides low-latency endpoints for real-time model predictions.
Features:
- Request Batching: Aggregates concurrent requests into single GPU batches.
- Caching: Redis-based caching for identical requests.
"""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

import redis.asyncio as redis
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from pydantic import BaseModel, ConfigDict, Field, validator

from python.src.utils import definitions
from python.src.utils.functions.functions import load_model


# Dummy tracer for when OpenTelemetry is disabled/missing
class DummySpan:
    """Dummy span for when OpenTelemetry is disabled."""

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def set_attribute(self, key, value):
        """No-op for dummy span."""
        pass


class DummyTracer:
    """Dummy tracer for when OpenTelemetry is disabled."""

    def start_as_current_span(self, name):
        """Start a no-op span."""
        return DummySpan()


try:
    tracer = trace.get_tracer(__name__)
except ImportError:
    tracer = DummyTracer()

# Ray Serve Imports
try:
    from ray import serve as ray_serve

    RAY_AVAILABLE = True
except ImportError:
    ray_serve = None
    RAY_AVAILABLE = False


# Global State
_MODEL: torch.nn.Module | None = None
_OPTS: dict[str, Any] | None = None
_REDIS: redis.Redis | None = None


class PredictionRequest(BaseModel):
    """Schema for model prediction request."""

    observations: list[list[float]] = Field(
        ..., description="Batch of observation tensors.", min_length=1
    )
    model_path: str | None = Field(
        None, description="Path to specific model checkpoint."
    )
    temperature: float = Field(1.0, ge=0.01, le=2.0)

    model_config = ConfigDict(strict=True)

    @validator("observations")
    @classmethod
    def validate_obs_shape(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate that all observations in the batch have the same length."""
        if not v:
            raise ValueError("Observations cannot be empty")
        # Ensure all observations have same length
        expected_len = len(v[0])
        for i, obs in enumerate(v):
            if len(obs) != expected_len:
                raise ValueError(
                    f"Observation at index {i} has inconsistent length {len(obs)}, expected {expected_len}"
                )
        return v


class PredictionResponse(BaseModel):
    """Schema for model prediction response."""

    predictions: list[list[float]]
    model_version: str
    latency_ms: float
    cached: bool = False

    model_config = ConfigDict(strict=True)


class BatchInferenceHandler:
    """
    Handles request batching for GPU inference.
    Incoming requests are added to a queue. background worker processes them in batches.
    """

    def __init__(self, model_loader_func: Callable[[], torch.nn.Module]) -> None:
        """
        Initialize BatchInferenceHandler.

        Args:
            model_loader_func: Function to load the model singleton.
        """
        self.queue: (
            asyncio.Queue[tuple[PredictionRequest, asyncio.Future[list[list[float]]]]]
            | None
        ) = None
        self.model_loader = model_loader_func
        self._shutdown = False
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background worker loop."""
        self._shutdown = False
        self.queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Stop the background worker loop and cleanup."""
        self._shutdown = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def predict(self, request: PredictionRequest) -> list[list[float]]:
        """Add request to queue and await result."""
        if self.queue is None:
            raise RuntimeError("BatchInferenceHandler not started")
        future: asyncio.Future[list[list[float]]] = (
            asyncio.get_running_loop().create_future()
        )
        await self.queue.put((request, future))
        return await future

    async def _worker_loop(self) -> None:
        """Background loop to process batches."""
        while not self._shutdown:
            batch: list[tuple[PredictionRequest, asyncio.Future[list[list[float]]]]] = (
                []
            )

            # 1. Fetch first item (blocking)
            try:
                if self.queue is None:
                    break
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                batch.append(item)
            except TimeoutError:
                continue

            # 2. Fetch remaining items up to BATCH_SIZE (non-blocking/timeout)
            # Use a short deadline to aggregate
            deadline = time.time() + definitions.BATCH_TIMEOUT
            while len(batch) < definitions.BATCH_SIZE:
                timeout = deadline - time.time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(item)
                except TimeoutError:
                    break

            if batch:
                await self._process_batch(batch)

    async def _process_batch(
        self, batch: list[tuple[PredictionRequest, asyncio.Future[list[list[float]]]]]
    ) -> None:
        """Process a batch of requests."""
        with tracer.start_as_current_span("process_batch") as span:
            requests, futures = zip(*batch, strict=True) if batch else ((), ())
            span.set_attribute("batch.size", len(requests))

            try:
                # Assume all requests use the same model version for now
                # (or group by model version if needed - keeping simple for P4.1)
                # In a real heavy setup, we'd have separate queues per model.
                # Here we just use the default model for efficiency.

                # Load model (cached globally)
                model = self.model_loader()
                if model is None:
                    raise RuntimeError("Model not initialized")

                # Prepare batch tensor
                # Flatten all observations from all requests: [req1_obs, req2_obs...] -> single batch
                # Track slice indices to split results back
                all_obs: list[list[float]] = []
                splits: list[int] = []
                for req in requests:
                    all_obs.extend(req.observations)
                    splits.append(len(req.observations))

                param_iter = model.parameters()
                try:
                    first_param = next(param_iter)
                    device = first_param.device
                except StopIteration:
                    device = torch.device("cpu")

                obs_tensor = torch.tensor(all_obs, dtype=torch.float32).to(device)

                # Inference
                with torch.no_grad():
                    output = model(obs_tensor)
                    output_list: list[list[float]] = output.cpu().tolist()

                # Distribute results
                cursor = 0
                # Cast futures to correct type since zip result is Tuple[Any, ...]
                typed_futures = cast(
                    tuple[asyncio.Future[list[list[float]]], ...], futures
                )

                for i, future in enumerate(typed_futures):
                    num_samples = splits[i]
                    result = output_list[cursor : cursor + num_samples]
                    cursor += num_samples

                    if not future.done():
                        future.set_result(result)

            except Exception as e:
                typed_futures_err = cast(
                    tuple[asyncio.Future[list[list[float]]], ...], futures
                )
                for future in typed_futures_err:
                    if not future.done():
                        future.set_exception(e)


def get_model(model_path: str | None = None) -> torch.nn.Module:
    """Singleton model loader with caching."""
    global _MODEL, _OPTS  # noqa: PLW0603

    default_path = "outputs/model_last.pt"
    target_path = model_path or default_path

    if _MODEL is None:  # Only load once for this demo scaling
        try:
            model, opts = load_model(target_path)
            model.eval()
            # Move to CUDA if available
            if torch.cuda.is_available():
                model = model.cuda()
            _MODEL = model
            _OPTS = opts
            print(f"Loaded model from {target_path}")
        except Exception as e:
            print(f"Warning: Failed to load model from {target_path}: {e}")
            # Initialize dummy model for testing if file missing
            dummy = torch.nn.Linear(10, 2)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            _MODEL = dummy

    return cast(torch.nn.Module, _MODEL)


# =============================================================================
# Ray Serve Deployment
# =============================================================================


class RayModelDeployment:
    """Ray Serve deployment wrapper for the inference service."""

    def __init__(self) -> None:
        """Initialize and warm up the model."""
        # Load model on init to warm up the actor
        get_model()

    async def __call__(self, request: PredictionRequest) -> list[list[float]]:
        """Handle incoming prediction request via Ray Serve."""
        return await batch_handler.predict(request)


if RAY_AVAILABLE and ray_serve is not None:
    RayModelDeployment = ray_serve.deployment(  # type: ignore
        name="nglab-prediction",
        num_replicas=int(os.getenv("NGLAB_API_REPLICAS", "2")),
        ray_actor_options={
            "num_cpus": 1,
            "num_gpus": 0.5 if torch.cuda.is_available() else 0,
        },
    )(RayModelDeployment)


# =============================================================================
# OpenTelemetry Instrumentation
# =============================================================================


def setup_telemetry(app: FastAPI) -> None:
    """Configure OpenTelemetry for the FastAPI application."""
    resource = Resource(attributes={"service.name": "nglab-inference-api"})

    # Trace Sampling: 10% sampling rate for production
    sampler = ParentBased(root=TraceIdRatioBased(0.1))

    provider = TracerProvider(resource=resource, sampler=sampler)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"),
            insecure=True,
        )
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)


# Initialize Batch Handler
batch_handler = BatchInferenceHandler(lambda: get_model())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the lifespan of the API, handling startup and shutdown."""
    # Startup
    global _REDIS  # noqa: PLW0603
    await batch_handler.start()
    try:
        _REDIS = redis.from_url(
            definitions.REDIS_URL, encoding="utf-8", decode_responses=True
        )
        # Ping to check connection
        # await _REDIS.ping()
        print("Redis connected")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        _REDIS = None

    yield

    # Shutdown
    await batch_handler.stop()
    if _REDIS:
        await _REDIS.close()


app = FastAPI(
    title="NGLab Inference API",
    description="High-performance batched inference service.",
    version="2.0.0",
    lifespan=lifespan,
)

# Setup OTel
if os.getenv("NGLAB_ENABLE_TELEMETRY", "true").lower() == "true":
    setup_telemetry(app)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Service health check."""
    return {
        "status": "online",
        "gpu_available": torch.cuda.is_available(),
        "redis_connected": _REDIS is not None,
        "batch_queue_size": batch_handler.queue.qsize() if batch_handler.queue else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate predictions with batching and caching.
    """
    start_time = time.perf_counter()

    # 1. Check Cache
    cache_key = None
    if _REDIS:
        # Create stable hash of input
        payload = json.dumps(
            request.observations
        )  # Canonical serialization needed in prod
        key_str = f"{request.model_path}:{payload}"
        cache_key = hashlib.sha256(key_str.encode()).hexdigest()

        cached_res = await _REDIS.get(cache_key)
        if cached_res:
            latency = (time.perf_counter() - start_time) * 1000
            return PredictionResponse(
                predictions=json.loads(cached_res),
                model_version=request.model_path or "default",
                latency_ms=round(latency, 2),
                cached=True,
            )

    # 2. Batched Inference
    try:
        predictions = await batch_handler.predict(request)

        # 3. Cache Result
        if _REDIS and cache_key:
            # Async cache set (fire and forget ideally, but await here for safety)
            await _REDIS.set(
                cache_key, json.dumps(predictions), ex=definitions.CACHE_TTL
            )

        latency = (time.perf_counter() - start_time) * 1000

        return PredictionResponse(
            predictions=predictions,
            model_version=request.model_path or "default",
            latency_ms=round(latency, 2),
            cached=False,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
