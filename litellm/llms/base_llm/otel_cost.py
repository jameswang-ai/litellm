from collections.abc import Mapping, Sequence
from math import isfinite, nan
from typing import Annotated, Final

from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import CacheCreationTokenDetails, ModelInfo, PromptTokensDetailsWrapper, Usage
from litellm.utils import get_model_info_helper


def _token_count(value: object) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError("Expected an unsigned token count")
    return int(value)


TokenCount = Annotated[int, BeforeValidator(_token_count), Field(ge=0, le=18446744073709551615)]


class TraceUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    model: str = Field(min_length=1, validation_alias=AliasChoices("gen_ai.response.model", "gen_ai.request.model"))
    provider: str | None = Field(default=None, validation_alias=AliasChoices("gen_ai.provider.name", "gen_ai.system"))
    input_tokens: TokenCount = Field(validation_alias="gen_ai.usage.input_tokens")
    output_tokens: TokenCount = Field(validation_alias="gen_ai.usage.output_tokens")
    cache_read_tokens: TokenCount = Field(default=0, validation_alias="gen_ai.usage.cache_read.input_tokens")
    cache_write_tokens: TokenCount = Field(default=0, validation_alias="gen_ai.usage.cache_write.input_tokens")
    cache_write_5m: TokenCount | None = Field(
        default=None, validation_alias="anthropic.usage.cache_creation.ephemeral_5m_input_tokens"
    )
    cache_write_1h: TokenCount | None = Field(
        default=None, validation_alias="anthropic.usage.cache_creation.ephemeral_1h_input_tokens"
    )
    service_tier: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "openai.response.service_tier", "anthropic.response.service_tier", "openai.request.service_tier"
        ),
    )

    @model_validator(mode="after")
    def consistent_cache_usage(self) -> "TraceUsage":
        if self.cache_read_tokens + self.cache_write_tokens > self.input_tokens:
            raise ValueError("Cached input exceeds total input")
        if self.cache_write_5m is not None or self.cache_write_1h is not None:
            if self.cache_write_5m is None or self.cache_write_1h is None:
                raise ValueError("Cache creation duration breakdown is incomplete")
            if self.cache_write_5m + self.cache_write_1h != self.cache_write_tokens:
                raise ValueError("Cache creation duration breakdown disagrees with total")
        return self

    def usage(self) -> Usage:
        details: Final = (
            CacheCreationTokenDetails(
                ephemeral_5m_input_tokens=self.cache_write_5m,
                ephemeral_1h_input_tokens=self.cache_write_1h,
            )
            if self.cache_write_5m is not None
            else None
        )
        return Usage(
            prompt_tokens=self.input_tokens,
            completion_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=self.cache_read_tokens,
                cache_write_tokens=self.cache_write_tokens,
                cache_creation_token_details=details,
            ),
        )


def estimate_cost(attributes: Mapping[str, str]) -> float | None:
    try:
        usage: Final = TraceUsage.model_validate(attributes)
        provider: Final = {"aws.bedrock": "bedrock", "gcp.vertex_ai": "vertex_ai"}.get(
            usage.provider or "", usage.provider
        )
        model_info: Final = get_model_info_helper(
            model=usage.model, custom_llm_provider=provider, allow_dynamic=False, default_token_cost=nan
        )
        if model_info["key"] not in litellm.model_cost:
            return None
        if any(
            ("audio" in key or "image" in key or "video" in key) and value != "0" for key, value in attributes.items()
        ):
            return None
        input_cost, output_cost = generic_cost_per_token(
            model=usage.model,
            custom_llm_provider=provider or model_info["litellm_provider"],
            usage=usage.usage(),
            service_tier=usage.service_tier,
            model_info=ModelInfo(**model_info, supported_openai_params=None),
            strict=True,
        )
        cost: Final = input_cost + output_cost
        return cost if isfinite(cost) and cost >= 0 else None
    except Exception:
        return None


def estimate_costs(attributes: Sequence[Mapping[str, str]]) -> tuple[float | None, ...]:
    return tuple(estimate_cost(item) for item in attributes)
