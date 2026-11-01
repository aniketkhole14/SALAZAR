"""Standalone entry point for the packaged SALAZAR backend."""

import os
import uvicorn

from app.api import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("SALAZAR_PORT", "8765")),
        log_level="warning",
    )
