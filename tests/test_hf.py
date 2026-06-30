import pytest

from aiod.hf import parse_repo_id


@pytest.mark.parametrize(
    "link,expected",
    [
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
        ("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/", "meta-llama/Llama-3.1-8B-Instruct"),
        ("https://huggingface.co/org/repo/tree/main", "org/repo"),
        ("https://huggingface.co/org/repo/resolve/main/config.json", "org/repo"),
        ("https://huggingface.co/org/repo?library=transformers", "org/repo"),
        ("https://huggingface.co/org/repo#usage", "org/repo"),
        ("http://huggingface.co/org/repo", "org/repo"),
        ("https://www.huggingface.co/org/repo", "org/repo"),
        ("https://hf.co/org/repo", "org/repo"),
        ("huggingface.co/org/repo", "org/repo"),
        ("org/repo@abc123", "org/repo"),
        ("  org/repo  ", "org/repo"),
    ],
)
def test_parse_repo_id_ok(link, expected):
    assert parse_repo_id(link) == expected


@pytest.mark.parametrize(
    "link",
    [
        "",
        "   ",
        "not a url",
        "single",
        "https://huggingface.co/spaces/org/repo",
        "https://huggingface.co/datasets/org/repo",
        "https://huggingface.co/models/org",
        "https://example.com/org/repo",
        "https://github.com/org/repo",
        "spaces/org/repo",
        "datasets/org/repo",
    ],
)
def test_parse_repo_id_raises(link):
    with pytest.raises(ValueError):
        parse_repo_id(link)


def test_parse_repo_id_rejects_non_string():
    with pytest.raises(ValueError):
        parse_repo_id(None)  # type: ignore[arg-type]
