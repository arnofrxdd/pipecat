#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam LLM service implementation using OpenAI-compatible interface."""

import json
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from openai import NOT_GIVEN

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.adapters.services.open_ai_adapter import is_given as openai_is_given
from pipecat.frames.frames import FunctionCallsFromLLMInfoFrame, LLMTextFrame
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import _NotGiven, assert_given, is_given
from pipecat.utils.tracing.service_decorators import traced_llm


@dataclass
class SarvamLLMSettings(OpenAILLMSettings):
    """Settings for SarvamLLMService.

    Parameters:
        wiki_grounding: Sarvam wiki grounding toggle.
        reasoning_effort: Reasoning effort level (low, medium, high).
    """

    wiki_grounding: bool | None | _NotGiven = field(default_factory=lambda: _NOT_GIVEN)
    reasoning_effort: Literal["low", "medium", "high"] | None | _NotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )


class SarvamLLMService(OpenAILLMService):
    """A service for interacting with Sarvam's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Sarvam's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Sarvam doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    _SUPPORTED_MODELS = frozenset(
        {
            "sarvam-105b",
            "sarvam-105b-conversations",
            "sarvam-105b-32k",
            "sarvam-30b",
            "sarvam-30b-16k",
            "glm-5-3",
            "glm5.3",
            "gemma-4-31b",
            "gemma4",
            "deepseek-v4.1-flash",
            "deepseekv4.1-flash",
        }
    )
    Settings = SarvamLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.sarvam.ai/v1",
        settings: Settings | None = None,
        default_headers: Mapping[str, str] | None = None,
        **kwargs,
    ):
        """Initialize Sarvam LLM service.

        Args:
            api_key: Sarvam API key used for both OpenAI auth and Sarvam subscription header.
            base_url: Sarvam OpenAI-compatible base URL.
            settings: Runtime-updatable settings.
            default_headers: Additional HTTP headers to include in requests.
            **kwargs: Additional keyword arguments passed to ``OpenAILLMService``.
        """
        # Initialize only Sarvam-specific defaults; inherited defaults are
        # provided by the OpenAI base service initialization.
        default_settings = self.Settings(
            model="sarvam-105b",
            wiki_grounding=None,
            reasoning_effort=None,
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        model = default_settings.model
        if not isinstance(model, str):
            raise ValueError("Sarvam LLM requires a non-empty model string.")
        self._validate_model(model)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            default_headers=default_headers,
            **kwargs,
        )

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create OpenAI-compatible client for Sarvam API endpoint.

        Ensures Sarvam auth and SDK identification headers are always attached.
        """
        merged_headers = dict(default_headers or {})
        # sdk_headers() carries Pipecat User-Agent and should override caller-provided value.
        merged_headers.update(sdk_headers())
        if api_key:
            merged_headers["api-subscription-key"] = api_key

        logger.debug(f"Creating Sarvam client with API {base_url}")
        return super().create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=merged_headers,
            **kwargs,
        )

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Sarvam chat completion request.

        Starts from OpenAI-compatible defaults, then removes unsupported
        request fields and applies Sarvam-specific options.
        """
        self._validate_tool_parameters(params_from_context)

        params = super().build_chat_completion_params(params_from_context)
        params.pop("stream_options", None)
        params.pop("max_completion_tokens", None)
        params.pop("service_tier", None)

        if is_given(self._settings.wiki_grounding) and self._settings.wiki_grounding is not None:
            params["wiki_grounding"] = self._settings.wiki_grounding
        if (
            is_given(self._settings.reasoning_effort)
            and self._settings.reasoning_effort is not None
        ):
            params["reasoning_effort"] = self._settings.reasoning_effort

        return params

    def _validate_model(self, model: str):
        if not model or not model.strip():
            raise ValueError("Sarvam LLM requires a non-empty model string.")
        if model not in self._SUPPORTED_MODELS:
            logger.warning(
                f"Sarvam LLM model '{model}' is not in known models list ({', '.join(sorted(self._SUPPORTED_MODELS))}). Proceeding anyway."
            )

    def _validate_tool_parameters(self, params_from_context: OpenAILLMInvocationParams):
        tools = params_from_context.get("tools", NOT_GIVEN)
        tool_choice = params_from_context.get("tool_choice", NOT_GIVEN)

        has_tools = (
            openai_is_given(tools)
            and tools is not None
            and (not isinstance(tools, list) or len(tools) > 0)
        )
        has_tool_choice = openai_is_given(tool_choice) and tool_choice is not None

        if has_tool_choice and not has_tools:
            raise ValueError("Sarvam requires non-empty `tools` when `tool_choice` is provided.")

    async def get_chat_completions(self, context: LLMContext):
        t0 = time.monotonic()
        adapter = self.get_llm_adapter()
        messages = (
            adapter.get_messages_for_logging(context)
            if hasattr(adapter, "get_messages_for_logging")
            else []
        )
        last_msg = messages[-1] if messages else {}
        last_role = last_msg.get("role") if isinstance(last_msg, dict) else "?"
        last_content = (
            last_msg.get("content") if isinstance(last_msg, dict) else str(last_msg)
        )

        params_from_context = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        params = self.build_chat_completion_params(params_from_context)
        model = params.get("model")
        temp = params.get("temperature")
        effort = params.get("reasoning_effort")
        tools = params.get("tools")
        tool_names = (
            [t.get("function", {}).get("name") for t in tools]
            if isinstance(tools, list)
            else []
        )

        logger.info(
            f"[DEBUG-TRACE: LLM] Request START -> model={model}, temp={temp}, "
            f"reasoning_effort={effort}, tools={tool_names}, "
            f"total_messages={len(messages)}, last_role={last_role}, "
            f"last_user_prompt={repr(str(last_content)[:140])}"
        )
        try:
            stream = await super().get_chat_completions(context)
            logger.info(
                f"[DEBUG-TRACE: LLM] HTTP stream connection response received in {(time.monotonic() - t0)*1000:.1f}ms"
            )
            return stream
        except Exception as e:
            logger.error(
                f"[DEBUG-TRACE: LLM] HTTP stream connection FAILED after {(time.monotonic() - t0)*1000:.1f}ms: {e}"
            )
            raise

    @traced_llm
    async def _process_context(self, context: LLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        self._pending_node_transition_function_calls = []
        text_generated_signal = False

        await self.start_ttfb_metrics()

        t_start = time.monotonic()
        chunk_stream = await self.get_chat_completions(context)

        @asynccontextmanager
        async def _closing(stream):
            chunk_iter = stream.__aiter__()
            try:
                yield chunk_iter
            finally:
                if hasattr(chunk_iter, "aclose"):
                    await chunk_iter.aclose()
                if hasattr(stream, "close"):
                    await stream.close()
                elif hasattr(stream, "aclose"):
                    await stream.aclose()

        chunk_count = 0
        first_chunk_time = None
        first_content_time = None
        reasoning_chars = 0
        content_chars = 0
        finish_reason = None
        tool_call_chunks = 0
        last_reasoning_log_time = t_start

        async with _closing(chunk_stream) as chunk_iter:
            async for chunk in chunk_iter:
                chunk_count += 1
                now = time.monotonic()
                if first_chunk_time is None:
                    first_chunk_time = now
                    logger.info(
                        f"[DEBUG-TRACE: LLM] FIRST chunk arrived from Sarvam in {(first_chunk_time - t_start)*1000:.1f}ms (TTFB)"
                    )

                if chunk.usage:
                    cached_tokens = (
                        chunk.usage.prompt_tokens_details.cached_tokens
                        if chunk.usage.prompt_tokens_details
                        else None
                    )
                    reasoning_tokens = (
                        chunk.usage.completion_tokens_details.reasoning_tokens
                        if chunk.usage.completion_tokens_details
                        else None
                    )
                    tokens = LLMTokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cache_read_input_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )
                    await self.start_llm_usage_metrics(tokens)

                if chunk.model and self.get_full_model_name() != chunk.model:
                    self.set_full_model_name(chunk.model)

                if chunk.choices is None or len(chunk.choices) == 0:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if not delta:
                    continue

                # Check for reasoning tokens/traces (e.g. reasoning_content)
                delta_reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if not delta_reasoning and hasattr(delta, "model_extra") and delta.model_extra:
                    delta_reasoning = delta.model_extra.get("reasoning_content") or delta.model_extra.get("reasoning")

                if delta_reasoning:
                    reasoning_chars += len(delta_reasoning)
                    if (now - last_reasoning_log_time) >= 1.5 or chunk_count <= 2:
                        last_reasoning_log_time = now
                        logger.info(
                            f"[DEBUG-TRACE: LLM: REASONING] Chunk #{chunk_count} (+{(now - t_start)*1000:.1f}ms): "
                            f"accumulated {reasoning_chars} reasoning chars | latest: {repr(delta_reasoning[:60])}"
                        )

                if delta.tool_calls:
                    tool_call_chunks += 1
                    tool_call = delta.tool_calls[0]
                    if tool_call.index != func_idx:
                        functions_list.append(function_name)
                        arguments_list.append(arguments or "{}")
                        tool_id_list.append(tool_call_id)
                        function_name = ""
                        arguments = ""
                        tool_call_id = ""
                        func_idx += 1
                    if tool_call.function and tool_call.function.name:
                        function_name += tool_call.function.name
                        tool_call_id = tool_call.id
                        logger.info(
                            f"[DEBUG-TRACE: LLM: TOOL CALL] Tool name: '{tool_call.function.name}' (+{(now - t_start)*1000:.1f}ms)"
                        )
                    if tool_call.function and tool_call.function.arguments:
                        arguments += tool_call.function.arguments

                elif delta.content:
                    if first_content_time is None:
                        first_content_time = now
                        await self.stop_ttfb_metrics()
                        logger.info(
                            f"[DEBUG-TRACE: LLM: FIRST CONTENT CHUNK] (+{(first_content_time - t_start)*1000:.1f}ms from stream start, "
                            f"reasoning_chars_before_content={reasoning_chars}): text={repr(delta.content)}"
                        )
                    content_chars += len(delta.content)
                    text_generated_signal = True
                    await self._push_llm_text(delta.content)

                elif (
                    hasattr(delta, "audio")
                    and delta.audio
                    and delta.audio.get("transcript")
                ):
                    await self.push_frame(LLMTextFrame(delta.audio["transcript"]))

        t_end = time.monotonic()
        total_duration_ms = (t_end - t_start) * 1000
        logger.info(
            f"[DEBUG-TRACE: LLM] Stream COMPLETED in {total_duration_ms:.1f}ms: "
            f"chunks={chunk_count}, content_chars={content_chars}, reasoning_chars={reasoning_chars}, "
            f"finish_reason={finish_reason}, "
            f"first_content_latency={(first_content_time - t_start)*1000 if first_content_time else -1:.1f}ms, "
            f"tool_call_chunks={tool_call_chunks}"
        )

        if content_chars == 0 and not function_name:
            logger.warning(
                f"[DEBUG-TRACE: LLM] DEAD AIR WARNING: LLM stream produced 0 content text and NO tool calls! "
                f"Total reasoning chars: {reasoning_chars}, finish_reason: {finish_reason}"
            )

        if function_name:
            functions_list.append(function_name)
            arguments_list.append(arguments or "{}")
            tool_id_list.append(tool_call_id)

            function_calls = []
            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(f"{self}: Failed to parse function call arguments: {arguments}")
                    continue
                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            logger.info(
                f"[DEBUG-TRACE: LLM] Dispatching function calls: {[fc.function_name for fc in function_calls]}"
            )
            await self.push_frame(
                FunctionCallsFromLLMInfoFrame(function_calls=function_calls),
                direction=FrameDirection.DOWNSTREAM,
            )
            await self._run_or_defer_function_calls(
                function_calls,
                text_generated=text_generated_signal,
            )

