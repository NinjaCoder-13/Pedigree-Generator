"""
llm_openrouter.py — asking any model on OpenRouter (e.g. DeepSeek V4 Flash)
===========================================================================

OpenRouter (https://openrouter.ai) gives one API, compatible with OpenAI's, for models from many
companies. `ask_openrouter()` sends the same prompt as the Claude path and returns a result in the
same shape as llm.ask_claude(), so the rest of the app treats both providers alike.

* SAME PROMPT, NO TOOLS: the system + user prompt from prompt.py, and no tools — the model can only
  reason from what it is shown (part of the no-leakage guarantee).
* STRUCTURED OUTPUT: the answer format (prompt.ANSWER_SCHEMA) is sent as a strict JSON Schema
  ("response_format"), so the reply is JSON in the requested shape.
* REPEATABILITY: temperature (default 0 = always pick the most likely next word) and a fixed seed
  are sent and recorded with the run. This makes repeated runs as similar as the model provider
  allows; it is not a hard guarantee, since providers' hardware can still differ slightly. Scoring
  itself is always deterministic (see scoring.py).
* ONLY PROVIDERS THAT HONOUR THE SETTINGS: OpenRouter may route a model to several hosting
  providers. "require_parameters" makes it use only providers that support every setting sent here
  (JSON schema, seed, temperature, reasoning effort), instead of silently ignoring some of them.
* REASONING: DeepSeek V4 and similar models think before answering. "Reasoning effort"
  (low / medium / high) sets how much. The thinking itself is not saved — only the final answer.
  Thinking counts toward "max output tokens", so allow plenty.
* STREAMING keeps the connection alive while the model thinks for a long time.
* KEY: from the OPENROUTER_API_KEY environment variable, or typed into the app. Never written to disk.
"""

import os
import time
from typing import Optional

import openai

from prompt import ANSWER_SCHEMA

BASE_URL = 'https://openrouter.ai/api/v1'

# Offered in the app; any other OpenRouter model ID can be typed in.
# "deepseek/deepseek-v4-flash" is DeepSeek V4 Flash. The dated name (-0731) pins one exact version,
# which is better for comparisons that must be repeatable months later.
MODELS = ['deepseek/deepseek-v4-flash', 'deepseek/deepseek-v4-flash-0731', 'deepseek/deepseek-v4.1-flash',
          'deepseek/deepseek-v4-pro']
EFFORTS = ['default', 'low', 'medium', 'high']   # 'default' = don't send a reasoning setting
DEFAULT_MAX_TOKENS = 64000


def api_key_from_env() -> Optional[str]:
    return os.environ.get('OPENROUTER_API_KEY')


def ask_openrouter(system: str, user: str, model: str, effort: str = 'default',
                   api_key: Optional[str] = None, temperature: float = 0.0, seed: int = 0,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """Send one request. Returns the same keys as llm.ask_claude():
         text, stop_reason ('end_turn' when finished normally, 'max_tokens' when cut off),
         served_by (the model and hosting provider that answered), usage, seconds, error"""
    started = time.time()
    key = api_key or api_key_from_env()
    if not key:
        return _failed(started, 'No OpenRouter key found. Type one in, or set OPENROUTER_API_KEY '
                                'before starting the app.')

    # OpenRouter-specific options go in "extra_body" (they are not part of OpenAI's own API).
    extra = {'provider': {'require_parameters': True}}
    if effort != 'default':
        extra['reasoning'] = {'effort': effort}

    try:
        client = openai.OpenAI(base_url=BASE_URL, api_key=key, timeout=1800)
        stream = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            response_format={'type': 'json_schema',
                             'json_schema': {'name': 'genotype_answers', 'strict': True, 'schema': ANSWER_SCHEMA}},
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            stream=True,
            stream_options={'include_usage': True},   # token counts arrive in the last chunk
            extra_body=extra,
        )
        # Collect the streamed answer piece by piece.
        parts, finish, served, usage = [], None, None, {}
        for chunk in stream:
            served = served or getattr(chunk, 'model', None)
            provider = getattr(chunk, 'provider', None) or (chunk.model_extra or {}).get('provider')
            if provider:
                served = f'{chunk.model} via {provider}'
            if chunk.usage:
                usage = dict(input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens)
            for choice in chunk.choices or []:
                if choice.delta and choice.delta.content:
                    parts.append(choice.delta.content)
                if choice.finish_reason:
                    finish = choice.finish_reason
    # Most specific errors first, so the message shown in the app says what to do.
    except openai.AuthenticationError:
        return _failed(started, 'OpenRouter rejected the key. Check OPENROUTER_API_KEY or the key you typed.')
    except openai.NotFoundError:
        return _failed(started, f'Model "{model}" was not found on OpenRouter, or no provider supports '
                                'all the settings (JSON schema, seed, temperature, reasoning).')
    except openai.RateLimitError:
        return _failed(started, 'Rate limit reached. Wait a minute and try again.')
    except openai.APIStatusError as e:
        hint = ' (402 usually means the OpenRouter account is out of credits)' if e.status_code == 402 else ''
        return _failed(started, f'OpenRouter error {e.status_code}{hint}: {e.message}')
    except openai.APIConnectionError:
        return _failed(started, 'Could not reach OpenRouter. Check the internet connection.')
    except openai.OpenAIError as e:
        return _failed(started, str(e))

    # Translate OpenAI-style finish reasons to the names the app uses for Claude.
    stop = {'stop': 'end_turn', 'length': 'max_tokens', 'content_filter': 'refusal'}.get(finish, finish)
    note = None
    if stop == 'max_tokens':
        note = 'The reply was cut off at the token limit (thinking counts too); raise "Max output tokens" ' \
               'or send one request per family.'
    elif stop == 'refusal':
        note = 'The provider filtered the reply.'
    elif finish is None:
        note = 'The stream ended without a finish reason; the reply may be incomplete.'
    return dict(text=''.join(parts), stop_reason=stop or 'unknown', served_by=served or model, usage=usage,
                seconds=round(time.time() - started, 1), error=note)


def _failed(started: float, message: str) -> dict:
    return dict(text='', stop_reason=None, served_by=None, usage={}, seconds=round(time.time() - started, 1),
                error=message)
