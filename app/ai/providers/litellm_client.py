"""LiteLLM provider — the only file allowed to import LiteLLM.

Responsibilities
----------------
* Configure LiteLLM via environment variables.
* Initialize LiteLLM Router with fallback, routing strategy, weights, and health checks.
* Return a LangChain-compatible ChatModel backed by LiteLLM Router.
"""

from __future__ import annotations

import os
from typing import Any, List
import yaml
from pathlib import Path

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from langchain_litellm import ChatLiteLLMRouter
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.messages.base import BaseMessage
import langsmith as ls

from app.core.config import settings
from app.core.logging import logger


import threading

_last_routing_info = threading.local()


class LiteLLMRoutingLogger(CustomLogger):
    """LiteLLM custom logger to track and log routing events (successes, fallbacks, failures)."""

    def log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        model_group = kwargs.get("model")
        actual_model = kwargs.get("litellm_params", {}).get("model")
        provider = actual_model.split("/")[0] if actual_model and "/" in actual_model else (actual_model or "unknown")
        latency_ms = (end_time - start_time).total_seconds() * 1000.0 if end_time and start_time else 0.0

        logger.info(
            "[LiteLLM Router] Routing success | Group: %s | Model: %s | Provider: %s | Latency: %.1fms",
            model_group,
            actual_model,
            provider,
            latency_ms,
        )

        try:
            usage = getattr(response_obj, "usage", None)
            if usage:
                # Can be a dict or a pydantic object/class
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)
                    total_tokens = getattr(usage, "total_tokens", 0)
            else:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

            _last_routing_info.model = actual_model or model_group or "unknown"
            _last_routing_info.prompt_tokens = prompt_tokens
            _last_routing_info.completion_tokens = completion_tokens
            _last_routing_info.total_tokens = total_tokens
        except Exception:
            pass

    def log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        model_group = kwargs.get("model")
        actual_model = kwargs.get("litellm_params", {}).get("model")
        exception = kwargs.get("exception")
        logger.warning(
            "[LiteLLM Router] Routing attempt failed | Group: %s | Model: %s | Error: %s",
            model_group,
            actual_model,
            exception,
        )

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


# Register global LiteLLM callbacks
litellm.callbacks = [LiteLLMRoutingLogger()]


def get_last_routing_info() -> dict:
    """Return the last successful routing event details from thread-local storage."""
    return {
        "model": getattr(_last_routing_info, "model", "unknown"),
        "prompt_tokens": getattr(_last_routing_info, "prompt_tokens", 0),
        "completion_tokens": getattr(_last_routing_info, "completion_tokens", 0),
        "total_tokens": getattr(_last_routing_info, "total_tokens", 0),
    }


class JobPatraRoutedChat(ChatLiteLLMRouter):
    """Custom ChatLiteLLMRouter that injects routing metadata to LangSmith dynamically."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: Any = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = super()._generate(messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs)
        self._inject_routing_metadata(res)
        return res

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: Any = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs)
        self._inject_routing_metadata(res)
        return res

    def _inject_routing_metadata(self, res: ChatResult) -> None:
        try:
            llm_output = res.llm_output or {}
            model_used = llm_output.get("model", "")
            
            if not model_used and res.generations:
                msg = res.generations[0].message
                if hasattr(msg, "response_metadata") and msg.response_metadata:
                    model_used = msg.response_metadata.get("model", "")

            # Set a fallback default if not found
            if not model_used:
                model_used = self.model

            # Determine provider
            provider = "unknown"
            if model_used:
                if "/" in model_used:
                    provider = model_used.split("/")[0]
                elif "gemini" in model_used.lower():
                    provider = "gemini"
                elif "groq" in model_used.lower() or "llama" in model_used.lower():
                    provider = "groq"
                else:
                    provider = model_used

            # Determine if fallback was used (compared to primary model)
            primary_model = "gemini/gemini-3.1-flash-lite"
            if self.router and self.router.model_list:
                primary_model = self.router.model_list[0].get("litellm_params", {}).get("model", primary_model)

            fallback_used = False
            if model_used:
                fallback_used = (model_used.lower() != primary_model.lower())

            # Get routing strategy
            routing_strategy = "priority"
            if self.router:
                routing_strategy = getattr(self.router, "routing_strategy", "priority")

            # Update active LangSmith run tree metadata
            rt = ls.get_current_run_tree()
            if rt:
                rt.metadata["provider"] = provider
                rt.metadata["model"] = model_used
                rt.metadata["fallback_used"] = fallback_used
                rt.metadata["routing_strategy"] = routing_strategy
        except Exception as exc:
            logger.warning("[Routing Logger] Failed to inject LangSmith metadata: %s", exc)


def load_router_config() -> dict[str, Any]:
    """Load and parse the router YAML configuration, overlaying environment variables."""
    config_path = Path(settings.LITELLM_ROUTER_CONFIG_PATH)
    if not config_path.exists():
        config_path = Path("config/litellm_router.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if settings.LITELLM_ROUTING_STRATEGY:
        config["routing_strategy"] = settings.LITELLM_ROUTING_STRATEGY

    # Inject timeout/retry overrides to all model deployments
    for model in config.get("model_list", []):
        params = model.setdefault("litellm_params", {})
        params["timeout"] = settings.LITELLM_TIMEOUT
        params["num_retries"] = settings.LITELLM_RETRY_COUNT

        # Resolve API keys from settings if present
        if "gemini" in params.get("model", "").lower() and settings.GOOGLE_API_KEY:
            params["api_key"] = settings.GOOGLE_API_KEY.get_secret_value()
        elif "groq" in params.get("model", "").lower() and settings.GROQ_API_KEY:
            params["api_key"] = settings.GROQ_API_KEY.get_secret_value()

    config["allowed_fails"] = settings.LITELLM_MAX_FAILURES
    config["cooldown_time"] = settings.LITELLM_COOLDOWN_TIME

    return config


def get_chat_model() -> ChatLiteLLMRouter:
    """Return a configured LangChain ChatLiteLLMRouter instance."""
    config = load_router_config()
    model_list = config.get("model_list", [])
    
    # LiteLLM does not support a "priority" routing strategy directly.
    # Priority-based failover is instead implemented via the "fallbacks" list and unique model names.
    strategy = config.get("routing_strategy", "simple-shuffle")


    # Extract router-specific settings
    router_settings = {
        "routing_strategy": strategy,
        "cooldown_time": float(config.get("cooldown_time", 30)),
        "allowed_fails": int(config.get("allowed_fails", 3)),
        "num_retries": int(config.get("num_retries", 2)),
    }

    # Initialize the LiteLLM Router with explicit fallbacks from config
    router = Router(
        model_list=model_list,
        fallbacks=config.get("fallbacks", []),
        **router_settings
    )

    model_alias = config.get("default_model_alias", "gemini-3.1-flash-lite")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    return JobPatraRoutedChat(
        router=router,
        model=model_alias,
        temperature=temperature,
        max_tokens=max_tokens,
    )
