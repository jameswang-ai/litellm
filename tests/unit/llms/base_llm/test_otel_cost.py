from collections.abc import Iterator, Mapping
from typing import Final

import pytest
import respx

import litellm
from litellm.llms.base_llm.otel_cost import estimate_cost, estimate_costs


@pytest.fixture
def usage(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mapping[str, str]]:
    monkeypatch.setitem(
        litellm.model_cost,
        "anthropic/trace-estimate-test",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
            "cache_read_input_token_cost": 0.1,
            "cache_creation_input_token_cost": 1.25,
            "cache_creation_input_token_cost_above_1hr": 2.0,
            "input_cost_per_token_priority": 3.0,
            "output_cost_per_token_priority": 4.0,
        },
    )
    yield {
        "gen_ai.response.model": "trace-estimate-test",
        "gen_ai.provider.name": "anthropic",
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": "10",
        "gen_ai.usage.cache_read.input_tokens": "60",
        "gen_ai.usage.cache_write.input_tokens": "20",
        "anthropic.usage.cache_creation.ephemeral_5m_input_tokens": "10",
        "anthropic.usage.cache_creation.ephemeral_1h_input_tokens": "10",
    }


def test_standard_cache_usage_uses_inclusive_input_and_existing_duration_pricing(usage: Mapping[str, str]) -> None:
    assert estimate_cost(usage) == pytest.approx(20 + 60 * 0.1 + 10 * 1.25 + 10 * 2 + 10 * 2)
    assert estimate_cost({**usage, "gen_ai.request.model": "unresolved-alias"}) == estimate_cost(usage)


@pytest.mark.parametrize(
    "field,value",
    (
        ("gen_ai.usage.input_tokens", ""),
        ("gen_ai.usage.output_tokens", "-1"),
        ("gen_ai.usage.output_tokens", "1.5"),
        ("gen_ai.usage.input_tokens", "79"),
        ("gen_ai.usage.cache_read.input_tokens", "bad"),
        ("anthropic.usage.cache_creation.ephemeral_1h_input_tokens", "11"),
        ("gen_ai.response.model", "unknown-trace-model"),
        ("gen_ai.usage.input_tokens.audio", "2"),
    ),
)
def test_invalid_or_unsupported_usage_remains_unknown(usage: Mapping[str, str], field: str, value: str) -> None:
    assert estimate_cost({**usage, field: value}) is None


def test_missing_usage_is_unknown_but_explicit_zero_is_priced(usage: Mapping[str, str]) -> None:
    identity: Final = {key: value for key, value in usage.items() if key in ("gen_ai.response.model", "gen_ai.provider.name")}
    assert estimate_cost(identity) is None
    assert estimate_cost({**identity, "gen_ai.usage.input_tokens": "0", "gen_ai.usage.output_tokens": "0"}) == 0


def test_served_tier_wins_over_requested_tier_and_invalid_span_does_not_lose_batch(usage: Mapping[str, str]) -> None:
    uncached: Final = {
        "gen_ai.response.model": usage["gen_ai.response.model"],
        "gen_ai.provider.name": "anthropic",
        "gen_ai.usage.input_tokens": "100",
        "gen_ai.usage.output_tokens": "10",
        "anthropic.response.service_tier": "priority",
        "openai.request.service_tier": "default",
    }
    assert estimate_costs((uncached, {}, usage)) == (340, None, estimate_cost(usage))


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize("provider", ("huggingface", "ollama", "ollama_chat", "lemonade", "deepseek"))
@pytest.mark.parametrize("prefixed", (False, True))
def test_untrusted_model_names_never_fetch_metadata(
    provider: str, prefixed: bool, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.route().respond(200, json={"max_position_embeddings": 8192})
    model: Final = "example/repo/resolve/main/large.bin#"
    attributes: Final = {
        "gen_ai.response.model": f"{provider}/{model}" if prefixed else model,
        "gen_ai.usage.input_tokens": "12",
        "gen_ai.usage.output_tokens": "3",
        **({} if prefixed else {"gen_ai.provider.name": provider}),
    }
    assert estimate_costs((attributes,)) == (None,)
    assert upstream.call_count == 0


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize("provider", ("huggingface", "ollama", "ollama_chat", "lemonade", "openai"))
@pytest.mark.parametrize("rate", (None, 0.0, 0.5))
def test_static_catalog_requires_declared_rates_without_dynamic_lookup(
    monkeypatch: pytest.MonkeyPatch, provider: str, rate: float | None, respx_mock: respx.MockRouter
) -> None:
    upstream: Final = respx_mock.route().respond(200, json={"max_position_embeddings": 8192})
    model: Final = f"{provider}/otel-catalog-test"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {"litellm_provider": provider, "mode": "chat", "input_cost_per_token": rate, "output_cost_per_token": 2.0},
    )
    actual: Final = estimate_cost(
        {"gen_ai.response.model": model, "gen_ai.usage.input_tokens": "10", "gen_ai.usage.output_tokens": "2"}
    )
    assert actual == (None if rate is None else 10 * rate + 4)
    assert upstream.call_count == 0


@pytest.mark.parametrize(
    "entry,input_tokens,service_tier,expected",
    (
        pytest.param(
            {"input_cost_per_token": "1e0", "output_cost_per_token": "2"}, 100, None, 120, id="numeric_strings"
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1, "output_cost_per_token": 2}]},
            100,
            None,
            120,
            id="tier_only_at_boundary",
        ),
        pytest.param(
            {
                "tiered_pricing": [
                    {"range": [0, 100], "input_cost_per_token": 1, "output_cost_per_token": 2},
                    {"range": [100, 200], "input_cost_per_token": 3, "output_cost_per_token": 4},
                ]
            },
            101,
            None,
            343,
            id="second_tier",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1, "output_cost_per_token": 2}]},
            101,
            None,
            121,
            id="above_last_tier",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1}], "output_cost_per_token": 2},
            100,
            None,
            120,
            id="tier_uses_flat_output",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100]}], "input_cost_per_token": 1, "output_cost_per_token": 2},
            100,
            None,
            120,
            id="unpriced_tier_uses_flat",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1}]},
            100,
            None,
            None,
            id="missing_tier_output",
        ),
        pytest.param(
            {"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1, "output_cost_per_token": 2}]},
            0,
            None,
            None,
            id="no_tier_for_zero_input",
        ),
        pytest.param({"tiered_pricing": []}, 100, None, None, id="empty_tiers"),
        pytest.param(
            {"input_cost_per_token_priority": 1, "output_cost_per_token_priority": 2},
            100,
            "priority",
            120,
            id="service_tier_only",
        ),
        pytest.param(
            {"input_cost_per_token_priority": 1, "output_cost_per_token_priority": 2},
            100,
            None,
            None,
            id="service_tier_not_selected",
        ),
        pytest.param(
            {"input_cost_per_token_above_100_tokens": 1, "output_cost_per_token_above_100_tokens": 2},
            101,
            None,
            121,
            id="threshold_only",
        ),
        pytest.param(
            {"input_cost_per_token_above_100_tokens": 1, "output_cost_per_token_above_100_tokens": 2},
            100,
            None,
            None,
            id="threshold_not_crossed",
        ),
        pytest.param(
            {"off_peak_pricing": {"hours_utc": "00:00-00:00", "input_cost_per_token": 1, "output_cost_per_token": 2}},
            100,
            None,
            120,
            id="active_off_peak_only",
        ),
        pytest.param(
            {"off_peak_pricing": {"hours_utc": [], "input_cost_per_token": 1, "output_cost_per_token": 2}},
            100,
            None,
            None,
            id="inactive_off_peak_only",
        ),
        pytest.param(
            {"input_cost_per_character": 1, "output_cost_per_token": 2}, 100, None, None, id="non_token_input"
        ),
    ),
)
def test_estimates_use_selected_rates_instead_of_requiring_flat_prices(
    monkeypatch: pytest.MonkeyPatch,
    entry: Mapping[str, object],
    input_tokens: int,
    service_tier: str | None,
    expected: float | None,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost, "openai/otel-rate-test", {"litellm_provider": "openai", "mode": "chat", **entry}
    )
    attributes: Final = {
        "gen_ai.response.model": "openai/otel-rate-test",
        "gen_ai.usage.input_tokens": str(input_tokens),
        "gen_ai.usage.output_tokens": "10",
        **({"openai.response.service_tier": service_tier} if service_tier is not None else {}),
    }
    assert estimate_cost(attributes) == expected


@pytest.mark.parametrize("invalid_rate", ("bad", -1.0, float("nan"), float("inf"), True, False))
@pytest.mark.parametrize("representation", ("flat", "tier", "priority", "threshold", "off_peak"))
def test_invalid_selected_rates_never_become_free_or_offset_another_charge(
    monkeypatch: pytest.MonkeyPatch, invalid_rate: str | float, representation: str
) -> None:
    rates: Final = {"input_cost_per_token": invalid_rate, "output_cost_per_token": 100}
    entries: Final = {
        "flat": rates,
        "tier": {"tiered_pricing": [{"range": [0, 1000], **rates}]},
        "priority": {"input_cost_per_token_priority": invalid_rate, "output_cost_per_token_priority": 100},
        "threshold": {
            "input_cost_per_token_above_10_tokens": invalid_rate,
            "output_cost_per_token_above_10_tokens": 100,
        },
        "off_peak": {
            "input_cost_per_token": 1,
            "output_cost_per_token": 100,
            "off_peak_pricing": {"hours_utc": "00:00-00:00", **rates},
        },
    }
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/otel-invalid-test",
        {"litellm_provider": "openai", "mode": "chat", **entries[representation]},
    )
    assert (
        estimate_cost(
            {
                "gen_ai.response.model": "openai/otel-invalid-test",
                "gen_ai.usage.input_tokens": "100",
                "gen_ai.usage.output_tokens": "10",
                "openai.response.service_tier": "priority",
            }
        )
        is None
    )


@pytest.mark.parametrize(
    "cache_rates,expected",
    (
        ({}, 120),
        (
            {
                "cache_read_input_token_cost": 0,
                "cache_creation_input_token_cost": 0,
                "cache_creation_input_token_cost_above_1hr": 0,
            },
            40,
        ),
        ({"cache_read_input_token_cost": "bad"}, None),
    ),
)
def test_tier_cache_rates_preserve_defaults_explicit_zero_and_invalid_values(
    monkeypatch: pytest.MonkeyPatch, usage: Mapping[str, str], cache_rates: Mapping[str, object], expected: float | None
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "anthropic/trace-estimate-test",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "tiered_pricing": [
                {"range": [0, 1000], "input_cost_per_token": 1, "output_cost_per_token": 2, **cache_rates}
            ],
        },
    )
    assert estimate_cost(usage) == expected


@pytest.mark.parametrize("tiered", (False, True))
@pytest.mark.parametrize(
    "write_rate,write_1h_rate,tokens_5m,tokens_1h,expected",
    (
        ("bad", 2, "0", "20", 140),
        (2, "bad", "20", "0", 140),
        (2, "bad", None, None, 140),
        ("bad", 2, "20", "0", None),
        (2, "bad", "0", "20", None),
        ("bad", 2, None, None, None),
    ),
)
def test_cache_write_rate_validation_follows_used_duration(
    monkeypatch: pytest.MonkeyPatch,
    usage: Mapping[str, str],
    tiered: bool,
    write_rate: str | int,
    write_1h_rate: str | int,
    tokens_5m: str | None,
    tokens_1h: str | None,
    expected: float | None,
) -> None:
    rates: Final = {
        "input_cost_per_token": 1,
        "output_cost_per_token": 2,
        "cache_creation_input_token_cost": write_rate,
        "cache_creation_input_token_cost_above_1hr": write_1h_rate,
    }
    monkeypatch.setitem(
        litellm.model_cost,
        "anthropic/trace-estimate-test",
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            **({"tiered_pricing": [{"range": [0, 1000], **rates}]} if tiered else rates),
        },
    )
    attributes: Final = {key: value for key, value in usage.items() if "ephemeral_" not in key}
    if tokens_5m is not None and tokens_1h is not None:
        attributes["anthropic.usage.cache_creation.ephemeral_5m_input_tokens"] = tokens_5m
        attributes["anthropic.usage.cache_creation.ephemeral_1h_input_tokens"] = tokens_1h
    assert estimate_cost(attributes) == expected


def test_unused_cache_rates_cannot_contaminate_uncached_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "openai/otel-uncached-test",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 1,
            "output_cost_per_token": 2,
            "cache_read_input_token_cost": "bad",
            "cache_creation_input_token_cost": "bad",
            "cache_creation_input_token_cost_above_1hr": "bad",
        },
    )
    assert estimate_cost(
        {
            "gen_ai.response.model": "openai/otel-uncached-test",
            "gen_ai.usage.input_tokens": "100",
            "gen_ai.usage.output_tokens": "10",
        }
    ) == 120
