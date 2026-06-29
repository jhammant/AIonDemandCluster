from aiod import profiles
from aiod.profiles import Profile


def test_builtins_present():
    assert "coder-7b" in profiles.BUILTIN
    assert profiles.get("coder-32b").model.startswith("Qwen/")


def test_from_dict_ignores_unknown_keys():
    p = Profile.from_dict("x", {"model": "org/m", "quant": "fp8", "bogus": 1})
    assert p.model == "org/m"
    assert p.quant == "fp8"
    assert not hasattr(p, "bogus")


def test_user_profile_overrides_and_roundtrips(tmp_path, monkeypatch):
    pf = tmp_path / "profiles.yaml"
    monkeypatch.setattr(profiles, "PROFILE_FILE", pf)

    # Override a built-in name + add a new one.
    profiles.save(Profile(name="coder-7b", model="me/custom-7b", quant="fp8"))
    profiles.save(Profile(name="mine", model="me/mine", provider="runpod", idle_minutes=15))

    allp = profiles.all_profiles()
    assert allp["coder-7b"].model == "me/custom-7b"  # user wins over built-in
    assert allp["mine"].provider == "runpod"
    assert profiles.is_builtin("coder-7b") is False  # now shadowed by user file
    assert profiles.is_builtin("coder-32b") is True


def test_remove_user_only(tmp_path, monkeypatch):
    pf = tmp_path / "profiles.yaml"
    monkeypatch.setattr(profiles, "PROFILE_FILE", pf)
    profiles.save(Profile(name="mine", model="me/mine"))
    assert profiles.remove("mine") is True
    assert profiles.remove("mine") is False
    # can't remove a pure built-in (not in user file)
    assert profiles.remove("coder-32b") is False
