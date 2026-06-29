from aiod import events


def test_events_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "EVENTS_FILE", tmp_path / "events.jsonl")
    assert events.read() == []
    assert events.latest() is None

    events.append("sizing", "org/model")
    events.append("renting", "1x H100 @ $2.50/hr")
    rows = events.read()
    assert [r["phase"] for r in rows] == ["sizing", "renting"]
    assert events.latest()["phase"] == "renting"
    assert events.latest()["msg"] == "1x H100 @ $2.50/hr"

    events.clear()
    assert events.read() == []


def test_read_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(events, "EVENTS_FILE", tmp_path / "events.jsonl")
    for i in range(30):
        events.append("loading", str(i))
    rows = events.read(5)
    assert len(rows) == 5
    assert rows[-1]["msg"] == "29"
