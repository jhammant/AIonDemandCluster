from aiod.watch import Activity, is_active, metrics_url, parse_metrics

SAMPLE = """
# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 2.0
vllm:num_requests_waiting{model_name="m"} 1.0
vllm:generation_tokens_total{model_name="m"} 100.0
vllm:generation_tokens_total{model_name="m2"} 50.0
"""


def test_metrics_url_strips_v1():
    assert metrics_url("http://1.2.3.4:8000/v1") == "http://1.2.3.4:8000/metrics"
    assert metrics_url("http://1.2.3.4:8000/v1/") == "http://1.2.3.4:8000/metrics"


def test_parse_metrics_sums_labels():
    m = parse_metrics(SAMPLE)
    assert m["vllm:num_requests_running"] == 2.0
    assert m["vllm:num_requests_waiting"] == 1.0
    assert m["vllm:generation_tokens_total"] == 150.0  # summed across labels


def test_is_active_when_requests_in_flight():
    assert is_active(Activity(in_flight=1, tokens=100), Activity(in_flight=0, tokens=100)) is True


def test_is_active_when_tokens_grew():
    assert is_active(Activity(in_flight=0, tokens=120), Activity(in_flight=0, tokens=100)) is True


def test_idle_when_quiet_and_no_growth():
    assert is_active(Activity(in_flight=0, tokens=100), Activity(in_flight=0, tokens=100)) is False
    assert is_active(Activity(in_flight=0, tokens=100), None) is False
