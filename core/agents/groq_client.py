"""
Shared resilient wrapper for Groq chat completions expected to return JSON.

The wrapper separates three failure classes because they require different
recovery strategies:

- connection failures -> retry with a longer exponential backoff;
- daily token quota exhaustion -> fail fast because retrying cannot help;
- transient rate limits or malformed/empty model output -> retry with an
  adjusted completion budget.

The rest of the application should not need to know about provider-specific
retry behavior or JSON extraction quirks.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from groq import APIConnectionError, APITimeoutError, Groq


class DailyQuotaExceededError(RuntimeError):
    """The provider's daily token quota has been exhausted."""


class GroqCallFailedError(RuntimeError):
    """The call failed after all configured retry attempts."""


CONNECTION_TROUBLESHOOTING = """Could not connect to api.groq.com after several
attempts. Check: 1) `curl -i https://api.groq.com`, 2) firewall/VPN/proxy,
3) HTTP_PROXY/HTTPS_PROXY environment variables, 4) SSL certificates
(`pip install -U certifi`)."""


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """
    Extract a JSON object from a response that contains surrounding text.

    Models occasionally wrap otherwise valid JSON in Markdown fences or
    additional prose. The extraction is intentionally conservative: only the
    outermost object is considered, and invalid JSON is treated as a parse
    failure so the caller can retry.
    """
    if not text:
        return None

    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Identify transient provider throttling from the provider error message.

    Groq can expose rate-limit conditions using different status/message
    combinations depending on the API path and failure mode.
    """
    message = str(exc).lower()

    return any(
        marker in message
        for marker in (
            "rate_limit_exceeded",
            "tokens per minute",
            "429",
        )
    )


def _is_daily_quota_error(exc: Exception) -> bool:
    """Identify a daily token quota exhaustion that cannot be fixed by retrying."""
    message = str(exc).lower()
    return "tokens per day" in message or "(tpd)" in message


def _is_connection_error(exc: Exception) -> bool:
    """Identify failures where retrying the network request is appropriate."""
    return isinstance(exc, (APIConnectionError, APITimeoutError))


def _call_with_connection_retry(
    client: Groq,
    request: dict[str, Any],
    max_retries: int,
) -> Any:
    """
    Execute a provider request with dedicated connection-level retries.

    Connection failures are handled separately from application-level retries
    because they do not consume a meaningful model-response attempt. A longer
    backoff is appropriate here, especially for temporary DNS, proxy, or
    network instability.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**request)
        except Exception as exc:
            if not _is_connection_error(exc):
                raise

            if attempt == max_retries:
                raise RuntimeError(
                    f"{CONNECTION_TROUBLESHOOTING}\n\n"
                    f"Last error: {exc}"
                ) from exc

            # Cap the backoff so a single task cannot block indefinitely.
            time.sleep(min(2**attempt, 20))

    raise AssertionError("Unreachable")


def robust_chat_json(
    client: Groq,
    model: str,
    messages: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    include_reasoning: bool = False,
    temperature: float = 0.2,
    max_completion_tokens: int = 1024,
    max_retries: int = 3,
    request_timeout: float = 60.0,
    connection_retries: int = 4,
) -> dict[str, Any]:
    """
    Call a Groq model and return a parsed JSON object.

    There are two retry layers:

    1. connection retries handle network-level failures;
    2. completion retries handle provider throttling and malformed model output.

    The completion budget is reduced after rate limits and increased after
    malformed/truncated output. This avoids repeatedly making the same request
    when the failure itself suggests that a different token budget is more
    appropriate.

    The final attempt deliberately omits JSON mode. Some reasoning models can
    produce an empty or truncated response when JSON mode is forced; allowing
    the final attempt to use the model's native output gives the parser a
    chance to recover a valid JSON object from the response.

    Raises:
        DailyQuotaExceededError: When the provider's daily token quota is
            exhausted. Retrying is intentionally skipped.
        GroqCallFailedError: When no valid JSON response is obtained after all
            completion retries.
        RuntimeError: When the provider remains unreachable after connection
            retries.
    """
    last_error = ""
    last_raw = ""
    current_budget = max_completion_tokens

    for attempt in range(1, max_retries + 1):
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": current_budget,
            "timeout": request_timeout,
            "include_reasoning": include_reasoning,
        }

        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort

        # Keep JSON mode on intermediate attempts. The final attempt is left
        # unconstrained so models that behave poorly under forced JSON mode
        # still have a chance to return parseable output.
        if attempt < max_retries:
            request["response_format"] = {"type": "json_object"}

        try:
            response = _call_with_connection_retry(
                client=client,
                request=request,
                max_retries=connection_retries,
            )
        except RuntimeError:
            # Connection troubleshooting is already complete and actionable.
            raise
        except Exception as exc:
            if _is_daily_quota_error(exc):
                raise DailyQuotaExceededError(
                    f"Daily token quota exhausted for model {model}. "
                    "Retrying will not help during this session.\n"
                    f"Details: {exc}"
                ) from exc

            if _is_rate_limit_error(exc):
                last_error = (
                    f"Rate limit (attempt {attempt}/{max_retries}, "
                    f"budget={current_budget}): {exc}"
                )

                # A smaller request is less likely to hit TPM limits again.
                current_budget = max(256, current_budget // 2)
                time.sleep(1.5 * attempt)
            else:
                last_error = (
                    f"{type(exc).__name__} "
                    f"(attempt {attempt}/{max_retries}): {exc}"
                )
                time.sleep(0.5 * attempt)

            continue

        message = response.choices[0].message

        # Some reasoning models place useful JSON in the reasoning field
        # instead of content. Treat that as a fallback rather than assuming
        # an empty content field means the request failed.
        raw = message.content or ""
        if not raw.strip():
            raw = getattr(message, "reasoning", "") or ""

        last_raw = raw

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        parsed = _extract_json_block(raw)
        if parsed is not None:
            return parsed

        last_error = (
            f"Invalid JSON (attempt {attempt}/{max_retries}), "
            f"raw[:200]={raw[:200]!r}"
        )

        # Malformed output often means the model was truncated. Give the next
        # attempt a larger completion budget, while keeping the growth bounded
        # indirectly by the caller's retry limit.
        current_budget = int(current_budget * 1.3)
        time.sleep(0.5 * attempt)

    raise GroqCallFailedError(
        f"Could not obtain valid JSON from model {model} after "
        f"{max_retries} attempts.\n"
        f"Last error: {last_error}\n"
        f"Last raw response: {last_raw[:500]!r}"
    )