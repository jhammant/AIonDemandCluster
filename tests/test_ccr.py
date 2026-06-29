import json

from aiod import ccr


def test_build_provider_full_path():
    p = ccr.build_provider("http://1.2.3.4:33526/v1", "sk-x", "org/model")
    assert p["api_base_url"] == "http://1.2.3.4:33526/v1/chat/completions"
    assert p["api_key"] == "sk-x"
    assert p["models"] == ["org/model"]
    assert p["name"] == ccr.PROVIDER_NAME


def test_router_refs_use_provider_comma_model():
    r = ccr.build_router("org/model")
    ref = f"{ccr.PROVIDER_NAME},org/model"
    assert r["default"] == ref
    assert r["background"] == ref
    assert r["think"] == ref
    assert r["longContext"] == ref
    assert r["webSearch"] == ref


def test_write_config_merges_and_preserves(tmp_path, monkeypatch):
    cfg_dir = tmp_path / ".claude-code-router"
    cfg_file = cfg_dir / "config.json"
    monkeypatch.setattr(ccr, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(ccr, "CONFIG_FILE", cfg_file)

    # Pre-existing config with another provider that must survive the merge.
    cfg_dir.mkdir(parents=True)
    cfg_file.write_text(
        json.dumps(
            {
                "LOG": True,
                "Providers": [{"name": "openrouter", "api_base_url": "x", "models": ["a"]}],
                "Router": {"default": "openrouter,a"},
            }
        )
    )

    ccr.write_config("http://5.6.7.8:40000/v1", "sk-y", "org/model")
    out = json.loads(cfg_file.read_text())

    names = {p["name"] for p in out["Providers"]}
    assert names == {"openrouter", ccr.PROVIDER_NAME}
    assert out["LOG"] is True  # preserved
    assert out["Router"]["default"] == f"{ccr.PROVIDER_NAME},org/model"
    assert cfg_file.with_suffix(".json.bak").exists()  # backed up
