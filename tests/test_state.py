from aiod import state


def test_download_progress_percent_and_phase():
    line = state.download_progress(135, 343, gpu_util=0)
    assert "135 / 343 GB" in line
    assert "(39%)" in line
    assert "downloading weights" in line


def test_download_progress_gpu_active_means_loading():
    line = state.download_progress(343, 343, gpu_util=42)
    assert "loading into VRAM" in line


def test_download_progress_complete_but_idle_gpu():
    line = state.download_progress(343, 343, gpu_util=0)
    assert "(100%)" in line
    assert "download complete" in line


def test_download_progress_caps_at_100():
    # disk_usage counts the image too, so it can exceed the weights size.
    line = state.download_progress(360, 343, gpu_util=0)
    assert "(100%)" in line


def test_download_progress_no_target_size():
    assert state.download_progress(80, None) == "80 GB on disk"


def test_download_progress_no_telemetry():
    assert state.download_progress(None, 343) is None


def test_load_tolerates_unknown_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state, "STATE_FILE", tmp_path / "state.json")
    inst = state.Instance(
        instance_id=1, repo_id="org/m", quant="bf16", gpu_desc="1x X",
        price_per_hr=1.0, created_at=0.0, ttl_hours=4.0, weights_gb=343.0,
    )
    state.save(inst)
    # Simulate a field written by a newer version.
    raw = (tmp_path / "state.json").read_text().rstrip().rstrip("}")
    (tmp_path / "state.json").write_text(raw + ',  "future_field": 99\n}')
    loaded = state.load()
    assert loaded is not None
    assert loaded.weights_gb == 343.0
