# AI on Demand Cluster (`aiod`)

Spin up any HuggingFace model on a **vast.ai** GPU, serve it with **vLLM**, and drive
it from **Claude Code** through **Claude Code Router (CCR)** — all from one command.

Give it a HuggingFace link; `aiod` figures out how much VRAM the model needs, finds
the cheapest matching machine on vast.ai, shows you the live **$/hr** before you commit,
rents it, waits for the model to load, and writes your CCR config so you can run
`ccr code` against your own hosted model.

```text
  HuggingFace link
        │  (params, dtype, KV shape, context)
        ▼
  ┌─────────────┐   estimate VRAM per quant      ┌──────────────────────┐
  │   sizing    │ ─────────────────────────────▶ │  live vast.ai offers  │  ← real $/hr
  └─────────────┘                                 └──────────────────────┘
        │  rent cheapest fit
        ▼
  vast.ai GPU box ──▶ vLLM (OpenAI-compatible /v1) ──▶ model weights
        ▲
        │  http://<public-ip>:<port>/v1/chat/completions  (+ bearer token)
        │
  Claude Code Router (local)  ◀── translates Anthropic ⇄ OpenAI
        ▲
        │
  Claude Code  (ccr code)
```

Why CCR sits in the middle: Claude Code speaks the **Anthropic** Messages API, while
vLLM serves the **OpenAI** API. CCR runs locally and translates between them, so the
remote box can stay a plain OpenAI-compatible server.

---

## Prerequisites

- **Python 3.10+**
- A **vast.ai** account ([sign up](https://cloud.vast.ai/?ref_id=25480)) + an API key from
  [Account → API Keys](https://cloud.vast.ai/manage-keys/) (required)
- A **HuggingFace token** — only for gated/private models like Llama/Gemma (optional)
- **Claude Code Router (classic CLI)** installed locally:

  ```bash
  npm install -g @musistudio/claude-code-router
  ```

  > Note: the project has a Desktop app (v3) that stores config under
  > `~/Library/Application Support/...`. `aiod` targets the **classic CLI**, whose
  > config lives at `~/.claude-code-router/config.json` and uses `ccr start/code/stop`.

## Install

> **Early release** — install from source for now. PyPI (`pipx install ai-on-demand`)
> is wired up and coming shortly.

```bash
git clone https://github.com/jhammant/AIonDemandCluster.git
cd AIonDemandCluster
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Then run `aiod init` to set up your keys.

<details>
<summary>Once on PyPI (pipx / uv / pip)</summary>

```bash
pipx install ai-on-demand     # or: uv tool install ai-on-demand  /  pip install ai-on-demand
```

</details>

<details>
<summary>Dev setup (tests + lint)</summary>

```bash
git clone https://github.com/jhammant/AIonDemandCluster.git
cd AIonDemandCluster
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q && ruff check aiod/
```

</details>

## Configure

**Guided setup (recommended)** — opens the right pages, validates each key as you paste
it, and writes `.env` for you:

```bash
aiod init
```

Check an existing setup any time:

```bash
aiod doctor
```

**Or configure manually:**

```bash
cp .env.example .env
# then edit .env:
#   VAST_API_KEY=...        (required)
#   HF_TOKEN=...            (optional — gated models only)
#   VLLM_API_KEY=           (optional — auto-generated per launch if blank)
#   AIOD_TTL_HOURS=4        (teardown reminder window)
#   AIOD_MAX_PRICE=6.0      (hard $/hr cap on offers)
```

`.env` is gitignored — secrets never reach the public repo. Keys are read from (in
order) **environment variables → project `.env` → global `~/.config/aiod/.env`**, so a
global install (`pipx install` / `uv tool install`) finds your keys from any directory.

---

## Usage

> **The TUI is the easiest way to drive everything** — start, estimate, launch, ping,
> and tear down from one screen: **`aiod tui`**. The CLI below is the scriptable
> equivalent (and what the TUI calls under the hood).

### Estimate cost before spending anything

```bash
aiod estimate Qwen/Qwen2.5-Coder-32B-Instruct
# or a full link:
aiod estimate https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
```

Prints the model's size and a table of **VRAM need / cheapest GPU fit / live $/hr** for
each quantization option (bf16, fp8, int4).

### Spin it up

```bash
aiod spin Qwen/Qwen2.5-Coder-32B-Instruct            # full precision (bf16)
aiod spin Qwen/Qwen2.5-Coder-32B-Instruct -q fp8     # ~half the VRAM
aiod spin <model> --max-price 3 --ttl 2 -y           # cap price, 2h window, no prompt
aiod spin <model> --idle 20                           # auto-shutdown after 20 idle min
aiod spin --profile coder-32b                         # use a saved preset (see below)
```

`spin` sizes the model, finds the cheapest fitting GPU, shows the plan, rents it, waits
for vLLM to finish loading, then writes your CCR config. When it's ready:

```bash
ccr restart && ccr code
```

…and Claude Code is now talking to your hosted model.

### Manage the instance

```bash
aiod status        # provider status, endpoint, cost so far, time left on the TTL
aiod ping --tools  # send a test prompt (+ tool call) to confirm it serves
aiod bench         # benchmark: TTFT, tokens/sec, throughput, $/1M tokens (add -c 8)
aiod watch --idle 20  # foreground idle watcher (auto-destroys when idle)
aiod ccr-config    # re-write the CCR config from the tracked instance
aiod teardown      # destroy the instance and stop billing  ← run this when done
```

### Engines — vLLM and llama.cpp (GGUF)

`aiod` auto-detects the model format: **safetensors / AWQ / fp8 → vLLM**, and
**GGUF → llama.cpp** (multi-part shards, multi-GPU, OpenAI endpoint). Force it with
`--engine vllm|llamacpp`, or pick it in the TUI. The right **tool-call parser** is
auto-selected per model family (a registry in `aiod/model_configs.py`) so function
calling works with Claude Code without hand-tuning flags.

```bash
# A 789B GGUF model on rented GPUs, benchmarked, for a couple of dollars:
aiod estimate huihui-ai/Huihui-GLM-5.2-abliterated-GGUF        # live $/hr
aiod spin     huihui-ai/Huihui-GLM-5.2-abliterated-GGUF -q UD-Q3_K_M --idle 30
aiod bench -c 4   # → tok/s + $/1M tokens
aiod teardown
```

### Profiles — named presets for spinning up a stack

A profile bundles model + quant + provider + price/idle settings under one name, so you
can launch a whole "architecture" by name (and pick it in the TUI).

```bash
aiod profile list                      # built-in + your presets
aiod profile show coder-32b
aiod profile add my-glm --model zai-org/GLM-4.6 --quant fp8 --idle 30 --max-price 4
aiod spin --profile my-glm             # launch it (any flag still overrides)
aiod profile path                      # the YAML file you can hand-edit
```

Profiles live in `~/.config/aiod/profiles.yaml` — adding a new one is just a YAML block.
Built-in starters: `coder-7b` (cheap test), `coder-32b`, `qwen3-coder-30b`, `glm-4.6`.

### Auto spin-up (on-demand) — the proxy

Run a local proxy and point Claude Code at it. The **first message spins the box up,
streams the live warm-up progress into the chat** (sizing → renting → booting →
downloading → ready), then continues with the real answer — all in one reply, no resend.
It auto-destroys on idle too.

```bash
aiod proxy --profile coder-32b --idle 20    # points CCR at the proxy automatically
ccr restart && ccr code                      # just start chatting — it spins on demand
```

Because progress streams continuously, the connection stays alive (no client timeout)
and you *see what's happening* instead of a silent hang. Watch it anywhere:
`aiod status`, the TUI, or `GET http://127.0.0.1:4000/aiod/status`. Non-streaming requests
(rare from Claude Code) get a short "warming up" reply to resend instead.

### Auto spin-down on idle

Pass `--idle N` to `spin` (or set it in a profile, or the TUI) and `aiod` starts a local
watcher that polls the box's vLLM metrics and **destroys the instance after N minutes with
no requests**. The TTL is a hard backstop. Note: the watcher runs on *your* machine, so if
it sleeps the watcher pauses — `aiod status` still shows when the TTL is blown.

### Interactive TUI (the control center)

```bash
aiod tui
```

One screen to run the whole lifecycle: pick a **profile** (or type a HuggingFace link),
choose quant / provider / idle window → **Estimate** live cost → **Launch** with streaming
progress → watch the **running-instance panel** (status, cost so far, idle/TTL, refreshed
live) → **Ping** or **Teardown**. CCR config is written automatically on launch.

---

## Choosing a model

Claude Code is **extremely tool-call heavy** — it lives on function calling, file edits,
and multi-step agentic loops. Pick models with strong tool-use support, e.g.:

| Model | Notes |
|---|---|
| `Qwen/Qwen2.5-Coder-32B-Instruct` | Great coding + tool calling; fits one 80GB GPU at int4/fp8 |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | MoE, fast, strong agentic behavior |
| `deepseek-ai/DeepSeek-V2.5` | Strong general + tool use (large) |
| `zai-org/GLM-4.6` | Solid agentic/coding model |

`aiod` sets `--enable-auto-tool-choice --tool-call-parser hermes` on vLLM, which suits
Qwen/Hermes-style models. Other families may need a different `--tool-call-parser`.

## Cost & safety

- **`--max-price` / `AIOD_MAX_PRICE`** — a hard ceiling; offers above it are never rented.
- **TTL** — `aiod status` shows time left and warns when exceeded. **Auto-destroy is a
  reminder, not enforced on the box** (putting your vast key on a public machine would be
  a security risk). **Always run `aiod teardown` when you're done** — billing continues
  until the instance is destroyed.
- The inference endpoint is protected by a bearer token (`--api-key`), but it is exposed
  on a public IP. Don't serve anything sensitive, and tear down promptly.

## How it works

- **`sizing.py`** — pulls model metadata from the HF Hub (params from safetensors, KV
  shape from `config.json`), estimates VRAM per quant (`weights × bytes + KV-cache +
  overhead`), and maps it to a GPU plan (count × tier, power-of-two tensor parallel).
- **`vast.py`** — vast.ai REST client: searches offers (filtering for
  `direct_port_count ≥ 1` so the port can be mapped), rents via `PUT /asks/{id}/`, reads
  the public `host:port` from `ports["8000/tcp"]`, and destroys on teardown.
- **`bootstrap.py`** — builds the vLLM launch (`vllm/vllm-openai` image + args: model,
  tensor-parallel size, quantization, api-key, HF token).
- **`health.py`** — polls `/v1/models` until the model is serving.
- **`ccr.py`** — merges an `aiod-vllm` provider + router into `~/.claude-code-router/config.json`
  (preserving your other providers; backs up the old config).

## Caveats

- **int4 (awq/gptq) needs a pre-quantized checkpoint** (a `*-AWQ` / `*-GPTQ` repo). On a
  full-precision repo, use `fp8` (works online) or `bf16`. `aiod` warns you.
- **Multi-GPU tensor parallel** uses NCCL over shared memory; some vast machines have a
  small `/dev/shm`. If a 2+ GPU launch crashes on `/dev/shm`, that's why — try a
  different host or a smaller quant that fits on one GPU.
- VRAM estimates are conservative (they assume concurrent sequences); real usage is often
  lower. Tune with `--concurrency` / `--context`.

## Providers

Pick the GPU backend with `--provider` (CLI) or the dropdown (TUI); set the matching key
in `.env`.

| Provider | Key | Notes |
|---|---|---|
| **vast.ai** (default) | `VAST_API_KEY` | Cheapest marketplace; bid on specific offers. |
| **RunPod** | `RUNPOD_API_KEY` | Clean API; uses a public **TCP** port (avoids the proxy's 100s stream timeout). |

```bash
aiod estimate <model> --provider runpod      # live RunPod pricing ($0)
aiod spin <model> --provider runpod -q fp8   # rent a RunPod pod
```

## Referral

Signup links use the maintainer's referral links so signing up through them supports the
project at **no extra cost to you**:

- **vast.ai** — `https://cloud.vast.ai/?ref_id=25480` (3% of referred spend, for the life of the account)
- **RunPod** — `https://runpod.io?ref=p8hj7fq3` (credits on referred spend)

Forking this repo? Point them at your own links in one place:
[`aiod/branding.py`](aiod/branding.py) (`VAST_REFERRAL_URL` / `RUNPOD_REFERRAL_URL`).

## License

MIT — see [LICENSE](LICENSE).
