from aiod.vast import extract_download_progress, recommend_disk_gb


def test_extract_progress_from_tqdm_line():
    log = (
        "Starting llama-server\n"
        "model-00001-of-00005.gguf:  47%|####6     | 3.2G/6.8G [00:12<00:14, 257MB/s]\n"
        "some other line"
    )
    p = extract_download_progress(log)
    assert p and "47%" in p and "3.2G/6.8G" in p


def test_extract_progress_picks_latest():
    log = "X: 10%| 1G/10G [.. 100MB/s]\nX: 80%| 8G/10G [.. 200MB/s]"
    assert "80%" in extract_download_progress(log)


def test_extract_progress_handles_carriage_returns():
    # tqdm overwrites the same line with \r — the freshest is last.
    log = "dl: 20%, 100MB/s\rdl: 95%, 100MB/s"
    assert "95%" in extract_download_progress(log)


def test_extract_progress_none_when_no_progress():
    assert extract_download_progress("") is None
    assert extract_download_progress("starting server\nloaded config") is None


def test_recommend_disk():
    assert recommend_disk_gb(343) >= 460  # GLM-5.2 Q3 needs lots of disk
    assert recommend_disk_gb(5) == 40  # floor
