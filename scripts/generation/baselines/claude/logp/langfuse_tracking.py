"""Optional Langfuse tracking for the Claude LogP batch benchmark."""

import os


def create_langfuse():
    """Create a Langfuse client if credentials are configured."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        print("Langfuse disabled: credentials are not configured.")
        return None

    try:
        from langfuse import Langfuse

        host = os.getenv(
            "LANGFUSE_HOST",
            "https://us.cloud.langfuse.com",
        )
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            environment=os.getenv(
                "LANGFUSE_ENVIRONMENT",
                "development",
            ),
            flush_at=1,
        )
    except Exception as exc:
        print(f"Warning: Langfuse initialization failed: {type(exc).__name__}")
        return None


def flush_langfuse(client):
    """Flush telemetry without interrupting benchmark execution."""
    if client is None:
        return

    try:
        client.flush()
    except Exception as exc:
        print(f"Warning: Langfuse flush failed: {type(exc).__name__}")


def record_observation(
    client,
    *,
    name,
    input_data,
    output_data,
    metadata=None,
    level=None,
    status_message=None,
    model=None,
    usage_details=None,
):
    """Record one observation; telemetry errors must not break the run."""
    if client is None:
        return

    try:
        observation = client.start_observation(
            name=name,
            as_type="generation" if model else "span",
            input=input_data,
            output=output_data,
            metadata=metadata or {},
            model=model,
            usage_details=usage_details,
            level=level,
            status_message=status_message,
        )
        observation.end()
    except Exception as exc:
        print(f"Warning: Langfuse observation failed: {type(exc).__name__}")
