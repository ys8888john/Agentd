"""配置读取与依赖组装。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .kernel.kernel import AgentKernel
from .kernel.llm import LLM, OllamaNativeLLM, OpenAICompatLLM


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    backend: str              # ollama | openai_compat | fake
    ollama_host: str
    ollama_model: str
    ollama_think: bool
    openai_base_url: str
    openai_model: str
    openai_api_key: str
    fake_reply: str
    system_prompt: str | None


def load_settings() -> Settings:
    return Settings(
        backend=_env("AGENTD_LLM_BACKEND", "ollama"),
        ollama_host=_env("AGENTD_OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=_env("AGENTD_OLLAMA_MODEL", "qwen3"),
        ollama_think=_env("AGENTD_OLLAMA_THINK", "false").lower() == "true",
        openai_base_url=_env("AGENTD_OPENAI_BASE_URL", "http://localhost:11434/v1"),
        openai_model=_env("AGENTD_OPENAI_MODEL", "qwen3"),
        openai_api_key=_env("AGENTD_OPENAI_API_KEY", "ollama"),
        fake_reply=_env("AGENTD_FAKE_REPLY", "这是 FakeLLM 的固定回复。"),
        system_prompt=os.getenv("AGENTD_SYSTEM_PROMPT") or None,
    )


def build_llm(settings: Settings | None = None) -> LLM:
    s = settings or load_settings()
    if s.backend == "ollama":
        return OllamaNativeLLM(
            host=s.ollama_host, model=s.ollama_model, think=s.ollama_think
        )
    if s.backend == "openai_compat":
        return OpenAICompatLLM(
            base_url=s.openai_base_url, model=s.openai_model, api_key=s.openai_api_key
        )
    if s.backend == "fake":
        from .kernel.llm import FakeLLM

        return FakeLLM(reply=s.fake_reply)
    raise ValueError(f"未知 LLM 后端: {s.backend}")


def build_kernel(settings: Settings | None = None) -> AgentKernel:
    s = settings or load_settings()
    return AgentKernel(llm=build_llm(s), system=s.system_prompt)
