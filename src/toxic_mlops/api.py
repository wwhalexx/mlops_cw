"""HTTP API: fail fast on missing artifacts, serve an immutable fitted pipeline."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from toxic_mlops import __version__
from toxic_mlops.config import Settings, load_settings
from toxic_mlops.model import load_model
from toxic_mlops.preprocessing import clean_text
from toxic_mlops.schemas import PredictResponse


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(
        min_length=1,
        max_length=10000,
        examples=["Спасибо за подробное объяснение, теперь всё понятно!"],
    )


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_version: str


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        resolved = settings or load_settings()
        model, metadata = load_model(resolved.inference.model_dir)
        if metadata["min_chars"] != resolved.inference.min_chars:
            raise ValueError("Input domain min_chars differs from trained artifact")
        app.state.settings = resolved
        app.state.model = model
        app.state.metadata = metadata
        yield
        app.state.model = None

    app = FastAPI(
        title="Russian toxic comments API",
        version=__version__,
        lifespan=lifespan,
        description="Binary classification; score is an uncalibrated model probability.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        if request.app.state.model is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return {
            "status": "ok",
            "model_ready": True,
            "model_version": request.app.state.metadata["model_version"],
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request):
        config = request.app.state.settings.inference
        if len(payload.text) > config.max_chars:
            raise HTTPException(status_code=422, detail="Text exceeds configured max_chars")
        if len(clean_text(payload.text)) < config.min_chars:
            raise HTTPException(
                status_code=422,
                detail=f"Text must contain at least {config.min_chars} characters after cleaning",
            )
        metadata = request.app.state.metadata
        score = float(request.app.state.model.predict_proba([payload.text])[0, 1])
        label = int(score >= metadata["threshold"])
        return {
            "label": label,
            "label_name": "toxic" if label else "non_toxic",
            "score": score,
            "threshold": metadata["threshold"],
            "model_version": metadata["model_version"],
            "preprocessing_version": metadata["preprocessing"],
        }

    return app


app = create_app()
