from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pricing.providers import negbeta as p
from tests.test_muna_provider import catalog as contract_catalog

MODEL = "qwen/qwen3.8-27b"


def catalog():
    payload = contract_catalog(MODEL)
    payload["provider"].update(
        id="negbeta", regions=["eu"], status_url="https://inference.negbeta.com",
        support_contact="https://inference.negbeta.com", incident_contact="https://inference.negbeta.com",
    )
    return payload


def test_discovers_prices_and_keeps_model_ids():
    prices, rows = p.discover(catalog())
    assert (prices[MODEL].prompt_micro_per_m, prices[MODEL].completion_micro_per_m) == (250000, 1750000)
    assert rows[MODEL]["upstream_id"] == MODEL


def test_rejects_empty_wrong_provider_and_wrong_contract():
    payload = catalog()
    payload["data"] = []
    with pytest.raises(RuntimeError):
        p.discover(payload)
    payload = catalog()
    payload["provider"]["id"] = "someone-else"
    with pytest.raises(RuntimeError, match="unexpected catalog provider"):
        p.discover(payload)
    payload = catalog()
    del payload["contract_version"]
    with pytest.raises(RuntimeError, match="Contract v2"):
        p.discover(payload)


def test_fetch_uses_key_and_canary_gates_routes(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "MANIFEST_PATH", tmp_path / "negbeta.json")
    monkeypatch.setenv("NEGBETA_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(p, "fetch_json", lambda *_a, **kw: calls.append(kw) or catalog())
    monkeypatch.setattr(p, "probe_openai_chat", lambda **kw: False)
    p.write_provider_manifest(p.fetch())
    assert calls[0]["extra_headers"] == {"Authorization": "Bearer test-key"}
    assert json.loads(p.MANIFEST_PATH.read_text())["models"][0]["routable"] is False
    monkeypatch.setattr(p, "probe_openai_chat", lambda **kw: True)
    p.write_provider_manifest(p.fetch())
    assert json.loads(p.MANIFEST_PATH.read_text())["models"][0]["routable"] is True
    monkeypatch.delenv("NEGBETA_API_KEY")
    with pytest.raises(RuntimeError, match="required"):
        p.fetch()


def test_registration_and_secret_wiring():
    from scripts.check_price_coverage import (
        _DISCOVERABLE_MANIFEST_PROVIDERS,
        _STALE_MANIFEST_PROVIDER_MODULE_BY_SLUG,
    )
    from scripts.pricing.refresh import PROVIDER_SLUGS
    from trusted_router.catalog_data import GATEWAY_PREPAID_PROVIDER_SLUGS, PROVIDERS
    from trusted_router.provider_manifest_policy import EXPIRING_PROVIDER_MANIFEST_SLUGS
    from trusted_router.providers import OPENAI_COMPATIBLE_PROVIDERS

    provider = PROVIDERS[p.SLUG]
    assert provider.provider_zero_data_retention and not provider.stores_content
    assert provider.supports_prepaid and not provider.supports_byok
    assert p.SLUG in GATEWAY_PREPAID_PROVIDER_SLUGS and p.SLUG in PROVIDER_SLUGS
    assert p.SLUG in EXPIRING_PROVIDER_MANIFEST_SLUGS
    assert _STALE_MANIFEST_PROVIDER_MODULE_BY_SLUG[p.SLUG] is p
    assert OPENAI_COMPATIBLE_PROVIDERS[p.SLUG] == (("NEGBETA_API_KEY",), p.BASE_URL)
    assert any(item[0] == p.SLUG for item in _DISCOVERABLE_MANIFEST_PROVIDERS)
    root = Path(__file__).resolve().parents[1]
    assert "NEGBETA_API_KEY:trustedrouter-negbeta-api-key" in (root / ".github/workflows/refresh-prices.yml").read_text()
    assert 'ensure_secret_from_env_file "NEGBETA_API_KEY" "trustedrouter-negbeta-api-key"' in (root / "scripts/deploy/secrets.sh").read_text()


def test_public_page(client):
    response = client.get("/providers/negbeta")
    assert response.status_code == 200
    for text in ("Negbeta", "EU", "https://inference.negbeta.com/docs", "https://negbeta.com/inference/dpa"):
        assert text in response.text
