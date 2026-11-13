"""Local MLflow traces with explicit metadata-only payloads."""

from pathlib import Path


def local_tracer(db: Path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{db.resolve()}")
    mlflow.set_experiment("billing-assistant-local")
    return lambda name, kind: mlflow.start_span(name=name, span_type=kind)
