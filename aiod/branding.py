"""Project links and the maintainer referral links.

To earn referral credit, paste YOUR referral link below for each provider.
  * vast.ai  -> Settings -> Referral Link   (3% of referred spend, for life)
  * RunPod   -> Settings -> Referrals       (credits on referred spend)

Leave an entry empty to fall back to the plain signup URL. These constants feed
the `aiod init` wizard and the README signup links, so you only edit them here.
"""

from __future__ import annotations

# >>> EDIT THESE: your referral links per provider <<<
VAST_REFERRAL_URL = "https://cloud.vast.ai/?ref_id=25480"
RUNPOD_REFERRAL_URL = "https://runpod.io?ref=p8hj7fq3"

# Plain signup fallbacks (used if the matching referral link above is empty).
SIGNUP_FALLBACK = {
    "vast": "https://cloud.vast.ai/",
    "runpod": "https://www.runpod.io/",
}
REFERRAL_URLS = {
    "vast": VAST_REFERRAL_URL,
    "runpod": RUNPOD_REFERRAL_URL,
}

# Other links.
VAST_KEYS_URL = "https://cloud.vast.ai/manage-keys/"
RUNPOD_KEYS_URL = "https://www.runpod.io/console/user/settings"
HF_TOKENS_URL = "https://huggingface.co/settings/tokens"
CCR_INSTALL_CMD = "npm install -g @musistudio/claude-code-router"


def signup_url(provider: str = "vast") -> str:
    """The link to send people who don't have an account on `provider` yet."""
    ref = (REFERRAL_URLS.get(provider) or "").strip()
    return ref or SIGNUP_FALLBACK.get(provider, SIGNUP_FALLBACK["vast"])
