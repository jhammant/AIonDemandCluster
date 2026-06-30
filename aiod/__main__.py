"""Enable ``python -m aiod`` (used by `aiod up --detach` to spawn the gateway)."""

from .cli import app

if __name__ == "__main__":
    app()
