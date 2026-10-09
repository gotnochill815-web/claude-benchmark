"""Optional Langfuse tracking for the Claude LogP batch benchmark."""

import os


def create_langfuse():
    """Return a Langfuse client when configured, otherwise disable tracking."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        print("Langfuse disabled: credentials are not configured.")
        return None

    from langfuse import Langfuse

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        environment=os.getenv("LANGFUSE_ENVIRONMENT", "development"),
        flush_at=1,
    )


def flush_langfuse(client):
    """Flush queued events without failing the benchmark on telemetry errors."""
    if client is None:
        return

    try:
        client.flush()
    except Exception as exc:
        print(f"Warning: Langfuse flush failed: {type(exc).__name__}")


def record_observation(client, *, name, input_data, output_data,
                       metadata=None, level=None, status_message=None,
                       model=None, usage_details=None):
    """Record one completed observation; telemetry must not break the run."""
    if client is None:
        return

    observation = None
    try:
        observation = client.start_observation(
            name=name,
            as_type="generation" if model else "span",
            input=input_data,
            metadata=metadata or {},
            model=model,
        )
        observation.update(
            output=output_data,
            metadata=metadata or {},
            level=level,
            status_message=status_message,
            usage_details=usage_details,
        )
    except Exception as exc:
        print(f"Warning: Langfuse observation failed: {type(exc).__name__}")
    finally:
        if observation is not None:
            try:
                observation.end()
            except Exception as exc:
                print(f"Warning: Langfuse finalize failed: {type(exc).__name__}")
