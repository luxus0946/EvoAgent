"""LLM client: direct OpenAI-compatible API plus a mock implementation (falls back when no key is set)."""

import json
import logging
import time
from abc import ABC, abstractmethod

import numpy as np

from evoagent.config import LLMConfig

logger = logging.getLogger("evoagent.llm")


class LLMError(Exception):
    """Exception raised for LLM call failures."""

    pass


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat_json(self, system: str, user: str, seed: int | None = None) -> dict:
        """Return the chat result as a JSON structure.

        Args:
            system: System prompt
            user: User prompt
            seed: Random seed (used by the mock implementation)

        Returns:
            Parsed JSON dictionary

        Raises:
            LLMError: If the call or parsing fails
        """
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    """OpenAI-compatible API client (DeepSeek, etc.)."""

    def __init__(self, config: LLMConfig | None = None):
        """Initialize.

        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        if not self.config.enabled:
            raise LLMError("DEEPSEEK_API_KEY 未配置")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    def chat_json(self, system: str, user: str, seed: int | None = None) -> dict:
        try:
            resp = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return _parse_json(content)
        except Exception as e:
            raise LLMError(f"LLM 调用失败: {e}") from e


class MockLLMClient(LLMClient):
    """Mock LLM client: generates a strategy from the prompt-genome fields by rules.

    Used for end-to-end validation and unit tests when no API key is set.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        """Initialize.

        Args:
            rng: Random number generator (used for parameter jitter; a fixed seed gives reproducibility)
        """
        self.rng = rng or np.random.default_rng(42)

    def chat_json(self, system: str, user: str, seed: int | None = None) -> dict:
        import re

        rng = np.random.default_rng(seed) if seed is not None else self.rng

        def _field(name: str, default: str) -> str:
            for line in user.splitlines():
                line = line.strip()
                if line.startswith(f"{name}:"):
                    value = line.split(":", 1)[1].strip()
                    match = re.match(r"[0-9.]+", value)
                    return match.group(0) if match else value
            return default

        def _float_field(name: str, default: float) -> float:
            return float(_field(name, str(default)))

        tool_pref = _field("工具偏好", "cma_es_first")
        exploration = _float_field("探索偏置", 0.5)
        thinking = _field("思维风格", "chain_of_thought")
        stopping = _float_field("收敛阈值", 0.3)

        pref_to_tool = {
            "cma_es_first": "cma_es",
            "bo_first": "bo",
            "ga_first": "ga",
            "ppo_first": "ppo",
            "diversify_first": "random_search",
        }
        initial = pref_to_tool.get(tool_pref, "cma_es")
        if exploration > 0.6:
            second = "random_search"
            switch = 0.6
        elif exploration < 0.3:
            second = "bo" if initial != "bo" else "cma_es"
            switch = 0.3
        else:
            second = "bo"
            switch = 0.5
        if thinking == "tree_of_thought":
            switch = min(0.9, switch + 0.3)
        elif thinking == "step_by_step":
            switch = max(0.1, switch - 0.1)

        jitter = 0.05 * rng.normal(size=2)
        return {
            "initial_tool": initial,
            "second_tool": second,
            "switch_after_ratio": float(np.clip(switch + jitter[0], 0.05, 0.95)),
            "stop_patience": float(np.clip(stopping + jitter[1], 0.05, 0.5)),
            "tool_params": {
                "cma_sigma": 0.25,
                "ga_mutation": 0.15,
                "sa_t0": 0.05,
                "sa_alpha": 0.995,
                "sa_sigma": 0.1,
                "bo_xi": 0.01,
                "ppo_lr": 0.01,
                "ppo_clip": 0.2,
                "ppo_gamma": 0.99,
            },
        }


def _parse_json(content: str) -> dict:
    """Extract JSON from LLM output (tolerates code fences and extra text)."""
    text = content.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError(f"LLM 输出不含 JSON: {content[:200]}")
    return json.loads(text[start : end + 1])


def build_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """Build an LLM client: use the real API when a key is set, otherwise fall back to the mock.

    Args:
        config: LLM configuration

    Returns:
        LLM client instance
    """
    config = config or LLMConfig()
    if config.enabled:
        try:
            return OpenAILLMClient(config)
        except LLMError as e:
            logger.warning("真实 LLM 初始化失败，回退模拟实现: %s", e)
    logger.info("使用模拟 LLM 客户端（未配置 DEEPSEEK_API_KEY）")
    return MockLLMClient()


def timed_chat(client: LLMClient, system: str, user: str, seed: int | None = None) -> dict:
    """LLM call with timing and logging."""
    t0 = time.perf_counter()
    result = client.chat_json(system, user, seed)
    logger.debug("LLM 调用耗时 %.2fs", time.perf_counter() - t0)
    return result
