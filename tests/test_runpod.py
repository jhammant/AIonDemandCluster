from aiod.bootstrap import ServerConfig
from aiod.runpod import RunpodClient, RunpodOffer


def test_endpoint_of_reads_publicip_and_portmappings():
    pod = {"publicIp": "213.173.109.39", "portMappings": {"8000": 13007}}
    assert RunpodClient.endpoint_of(pod, 8000) == ("213.173.109.39", 13007)


def test_endpoint_of_none_until_placed():
    assert RunpodClient.endpoint_of({"publicIp": None, "portMappings": None}, 8000) is None
    assert RunpodClient.endpoint_of({"publicIp": "1.2.3.4", "portMappings": {}}, 8000) is None


def test_status_of_reads_desired_status():
    assert RunpodClient.status_of({"desiredStatus": "RUNNING"}) == "RUNNING"
    assert RunpodClient.status_of({}) == "unknown"


def test_build_create_body_uses_tcp_and_docker_start_cmd():
    cfg = ServerConfig(repo_id="org/m", num_gpus=2, quant="bf16", api_key="sk-x", hf_token="hf_y")
    body = RunpodClient.build_create_body(
        cfg, disk_gb=60, max_price=5.0, label="aiod", gpu_type_id="NVIDIA H100 80GB HBM3"
    )
    assert body["ports"] == ["8000/tcp"]  # public TCP, not the proxy URL
    assert body["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3"]
    assert body["gpuCount"] == 2
    assert body["cloudType"] == "SECURE"
    assert "--model" in body["dockerStartCmd"]
    assert body["env"]["HF_TOKEN"] == "hf_y"


def test_offer_desc():
    o = RunpodOffer(id="x", gpu_name="H100 80GB", num_gpus=2, dph_total=5.0, total_vram_gb=160)
    assert o.desc == "2x H100 80GB"


def test_requires_api_key():
    import pytest

    from aiod.runpod import RunpodError

    with pytest.raises(RunpodError):
        RunpodClient("")
