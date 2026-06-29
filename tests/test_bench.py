from aiod.bench import BenchResult, ReqResult, _pct


def test_percentile():
    assert _pct([], 50) is None
    assert _pct([1.0], 50) == 1.0
    assert _pct([1, 2, 3, 4, 5], 50) == 3
    assert _pct([1, 2, 3, 4, 5], 95) == 5


def test_decode_speed():
    r = ReqResult(ok=True, ttft=0.5, total=2.5, completion_tokens=200)
    # 200 tokens over (2.5 - 0.5)=2s -> 100 tok/s
    assert abs(r.decode_tok_s - 100.0) < 1e-6


def test_aggregate_throughput_and_cost():
    # 4 requests, 100 tokens each = 400 tokens over 2s wall -> 200 tok/s.
    rs = [ReqResult(ok=True, ttft=0.2, total=1.5, completion_tokens=100) for _ in range(4)]
    res = BenchResult(n=4, concurrency=4, wall_time=2.0, results=rs, price_per_hr=4.27)
    assert res.total_completion_tokens == 400
    assert abs(res.throughput_tok_s - 200.0) < 1e-6
    # $/1M = (4.27/3600)/200 * 1e6 ≈ 5.93
    assert abs(res.cost_per_million - 5.93) < 0.05


def test_failures_excluded():
    rs = [ReqResult(ok=True, ttft=0.2, total=1.0, completion_tokens=50),
          ReqResult(ok=False, error="boom")]
    res = BenchResult(n=2, concurrency=1, wall_time=1.0, results=rs)
    assert len(res.ok) == 1
    assert res.total_completion_tokens == 50
    assert res.cost_per_million is None  # no price set
