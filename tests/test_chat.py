import importlib.resources

from starlette.testclient import TestClient

from aiod import proxy, state
from aiod.config import Settings


def _settings():
    return Settings(
        vast_api_key="k", hf_token=None, vllm_api_key="sk-aiod-secret", ttl_hours=4, max_price=6.0
    )


def _manager(monkeypatch, *, require_auth=False):
    monkeypatch.setattr(state, "load", lambda: None)
    m = proxy.Manager(
        _settings(),
        {"model": "org/cool-model", "quant": "fp8"},
        idle_minutes=20,
        require_auth=require_auth,
    )
    m.spinning = True
    return m


# --------------------------------------------------------------------------- #
# chat_page rendering + routing
# --------------------------------------------------------------------------- #

def test_chat_page_returns_html_with_model_replaced(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch))
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "org/cool-model" in r.text
    assert "__AIOD_MODEL__" not in r.text


def test_chat_route_also_serves_page(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch))
    with TestClient(app) as client:
        root = client.get("/")
        chat = client.get("/chat")
    assert root.status_code == 200
    assert chat.status_code == 200
    # Same rendered page from both routes.
    assert chat.text == root.text
    assert "org/cool-model" in chat.text


def test_token_injected_on_loopback(monkeypatch):
    app = proxy.build_app(_manager(monkeypatch, require_auth=False))
    with TestClient(app) as client:
        r = client.get("/")
    assert "sk-aiod-secret" in r.text
    assert "__AIOD_TOKEN__" not in r.text


def test_token_not_leaked_when_require_auth(monkeypatch):
    """No-leak: on a non-loopback / require_auth bind the bearer token must NOT
    be embedded in the served HTML."""
    app = proxy.build_app(_manager(monkeypatch, require_auth=True))
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "sk-aiod-secret" not in r.text
    assert "__AIOD_TOKEN__" not in r.text  # placeholder still replaced (with empty)


# --------------------------------------------------------------------------- #
# webui_docker_cmd
# --------------------------------------------------------------------------- #

def test_webui_docker_cmd_wires_gateway_and_token():
    cmd = proxy.webui_docker_cmd(4000, 3000, "sk-aiod-secret")
    joined = " ".join(cmd)
    assert "OPENAI_API_BASE_URL=http://host.docker.internal:4000/v1" in cmd
    assert "OPENAI_API_KEY=sk-aiod-secret" in cmd
    assert "WEBUI_AUTH=False" in cmd
    # -p host:container mapping
    i = cmd.index("-p")
    assert cmd[i + 1] == "3000:8080"
    assert "--name" in cmd and "aiod-openwebui" in cmd
    assert cmd[-1] == "ghcr.io/open-webui/open-webui:main"
    assert "docker run -d" in joined


def test_webui_docker_cmd_custom_ports():
    cmd = proxy.webui_docker_cmd(5005, 8888, "tok")
    assert "OPENAI_API_BASE_URL=http://host.docker.internal:5005/v1" in cmd
    i = cmd.index("-p")
    assert cmd[i + 1] == "8888:8080"


# --------------------------------------------------------------------------- #
# package data is readable via importlib.resources (installed-build contract)
# --------------------------------------------------------------------------- #

def test_chat_html_readable_via_importlib_resources():
    html = (importlib.resources.files("aiod") / "web" / "chat.html").read_text(encoding="utf-8")
    assert "__AIOD_MODEL__" in html
    assert "__AIOD_TOKEN__" in html
    assert "/v1/chat/completions" in html
