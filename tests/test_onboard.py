from aiod import onboard


def test_read_env_skips_comments_and_blanks(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\n\nVAST_API_KEY=abc123\nHF_TOKEN = tok \n")
    env = onboard.read_env(p)
    assert env == {"VAST_API_KEY": "abc123", "HF_TOKEN": "tok"}


def test_set_env_values_updates_in_place_preserving_comments(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# my secrets\nVAST_API_KEY=old\nAIOD_TTL_HOURS=4\n")
    onboard.set_env_values({"VAST_API_KEY": "new"}, path=p)
    text = p.read_text()
    assert "# my secrets" in text  # comment preserved
    assert "VAST_API_KEY=new" in text
    assert "old" not in text
    assert "AIOD_TTL_HOURS=4" in text  # untouched


def test_set_env_values_appends_missing_keys(tmp_path):
    p = tmp_path / ".env"
    p.write_text("VAST_API_KEY=abc\n")
    onboard.set_env_values({"HF_TOKEN": "tok", "AIOD_MAX_PRICE": "3.0"}, path=p)
    env = onboard.read_env(p)
    assert env["VAST_API_KEY"] == "abc"
    assert env["HF_TOKEN"] == "tok"
    assert env["AIOD_MAX_PRICE"] == "3.0"


def test_validate_hf_token_empty_is_ok_offline():
    ok, msg = onboard.validate_hf_token("")
    assert ok is True
    assert "not set" in msg


def test_validate_vast_key_empty_offline():
    ok, msg = onboard.validate_vast_key("")
    assert ok is False
