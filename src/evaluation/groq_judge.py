"""
Groq (qwen/qwen3-32b) wrapped as a DeepEval-compatible judge model.

History: Gemini hit an undocumented 20/day account quota; Groq's
llama-3.1-8b-instant failed at reliable structured output; qwen3-32b's
tool-calling mode failed to parse its own valid JSON (fixed via JSON
mode instead); qwen3-32b's default reasoning mode then blew the TPM
budget (fixed via reasoning_effort="none"). This version adds one more
layer: instructor's own max_retries only retries on SCHEMA VALIDATION
failures, not transport-level errors (429/503/connection issues) --
confirmed by those errors consistently showing "Total attempts: 1"
despite max_retries=3 being set. The retry helpers below handle that
gap explicitly with exponential backoff.
"""

import os
import time
import asyncio
import random
from typing import Optional, Type
import instructor
from groq import Groq, AsyncGroq
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

GROQ_JUDGE_MODEL = "qwen/qwen3-32b"

_sync_client = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY")), mode=instructor.Mode.JSON
)
_async_client = instructor.from_groq(
    AsyncGroq(api_key=os.getenv("GROQ_API_KEY")), mode=instructor.Mode.JSON
)

MIN_INTERVAL_SECONDS = 1.2
_last_reserved_time = 0.0


async def _throttle_async():
    global _last_reserved_time
    now = time.monotonic()
    wait = MIN_INTERVAL_SECONDS - (now - _last_reserved_time)
    _last_reserved_time = max(now, _last_reserved_time + MIN_INTERVAL_SECONDS)
    if wait > 0:
        await asyncio.sleep(wait)


def _throttle_sync():
    global _last_reserved_time
    now = time.monotonic()
    wait = MIN_INTERVAL_SECONDS - (now - _last_reserved_time)
    _last_reserved_time = max(now, _last_reserved_time + MIN_INTERVAL_SECONDS)
    if wait > 0:
        time.sleep(wait)


def _with_backoff_sync(fn, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = min(30, (2 ** attempt) + random.uniform(0, 1))
            print(f"  [judge] {type(e).__name__}, retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_attempts})...")
            time.sleep(wait)


async def _with_backoff_async(coro_fn, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return await coro_fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = min(30, (2 ** attempt) + random.uniform(0, 1))
            print(f"  [judge] {type(e).__name__}, retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_attempts})...")
            await asyncio.sleep(wait)


class GroqJudge(DeepEvalBaseLLM):
    def load_model(self):
        return _sync_client

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        _throttle_sync()
        client = self.load_model()
        if schema is not None:
            return _with_backoff_sync(lambda: client.chat.completions.create(
                model=GROQ_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                max_retries=3,
                reasoning_effort="none",
            ))
        response = _with_backoff_sync(lambda: client.chat.completions.create(
            model=GROQ_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="none",
        ))
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        await _throttle_async()
        if schema is not None:
            return await _with_backoff_async(lambda: _async_client.chat.completions.create(
                model=GROQ_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                max_retries=3,
                reasoning_effort="none",
            ))
        response = await _with_backoff_async(lambda: _async_client.chat.completions.create(
            model=GROQ_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="none",
        ))
        return response.choices[0].message.content

    def get_model_name(self) -> str:
        return "Groq qwen/qwen3-32b (DeepEval judge, via instructor)"