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
  (JSON schema, seed, temperature, reasoning), instead of silently ignoring some of them.
* REASONING IS ALWAYS ON: reasoning is always switched on explicitly ("default" effort = on, at the
  model's own default effort), so no host can answer without thinking. A reply that comes back
  with no reasoning is not accepted: the request is retried with that host excluded, up to
  MAX_ATTEMPTS times. How many reasoning tokens were used is recorded with the run. The thinking
  text itself is not saved — only the final answer.
* NO TOKEN LIMIT: no "max_tokens" is sent, so the model can think and answer as long as the host
  allows (its own maximum output length).
* STREAMING keeps the connection alive while the model thinks for a long time.
* COST: OpenRouter reports what each request cost (in US dollars) with the token counts. If a
  host leaves it out, it is looked up afterwards from OpenRouter's generation record. Attempts that
  were rejected for having no reasoning were still paid for, so their cost is added in.
* KEY: from the OPENROUTER_API_KEY environment variable, or typed into the app. Never written to disk.
"""

import json
import os
import time
import urllib.request
from typing import Callable, Optional

import openai

from prompt import ANSWER_SCHEMA

BASE_URL = 'https://openrouter.ai/api/v1'

# Offered in the app; any other OpenRouter model ID can be typed in.
# "deepseek/deepseek-v4-flash" is DeepSeek V4 Flash. The dated name (-0731) pins one exact version,
# which is better for comparisons that must be repeatable months later.
MODELS = ['deepseek/deepseek-v4-flash', 'deepseek/deepseek-v4-flash-0731', 'deepseek/deepseek-v4.1-flash',
          'deepseek/deepseek-v4-pro']
EFFORTS = ['default', 'low', 'medium', 'high']   # 'default' = reasoning on, at the model's default effort
MAX_ATTEMPTS = 3        # tries when a host replies without reasoning (each retry excludes that host)


def api_key_from_env() -> Optional[str]:
    return os.environ.get('OPENROUTER_API_KEY')


def ask_openrouter(system: str, user: str, model: str, effort: str = 'default',
                   api_key: Optional[str] = None, temperature: float = 0.0, seed: int = 0,
                   on_text: Optional[Callable[[str], None]] = None) -> dict:
    """Send one request. Returns the same keys as llm.ask_claude():
         text, stop_reason ('end_turn' when finished normally, 'max_tokens' when cut off),
         served_by (the model and hosting provider that answered), usage (incl. reasoning_tokens),
         seconds, error
    A reply without any reasoning is rejected and retried with that host excluded.
    `on_text`, if given, is called with each new piece of the answer as it streams in, and with None
    when a rejected attempt is retried (so whatever it collected so far should be thrown away)."""
    started = time.time()
    key = api_key or api_key_from_env()
    if not key:
        return _failed(started, 'No OpenRouter key found. Type one in, or set OPENROUTER_API_KEY '
                                'before starting the app.')
    excluded, spent = [], 0.0               # spent = cost of earlier, rejected attempts
    for attempt in range(MAX_ATTEMPTS):
        if attempt and on_text:
            on_text(None)
        res = _ask_once(system, user, model, effort, key, temperature, seed, excluded, started, on_text)
        host = res.pop('provider', None)
        if res.get('cost_usd') is not None:
            res['cost_usd'] = round(res['cost_usd'] + spent, 6)
        spent = res.get('cost_usd') or spent
        if res['stop_reason'] is None or res.pop('reasoned'):
            if excluded and res['stop_reason'] is not None:
                res['error'] = (res['error'] + ' ' if res['error'] else '') + \
                    f'Skipped host(s) that answered without reasoning: {", ".join(excluded)}.'
            return res
        excluded.append(host or 'unknown')
        if host is None:        # can't tell which host it was, so can't exclude it
            break
    return _failed(started, 'Every attempt came back without reasoning (host(s): '
                            f'{", ".join(excluded)}). Nothing was accepted; try again later or another model.')


def _ask_once(system: str, user: str, model: str, effort: str, key: str, temperature: float, seed: int,
              excluded: list, started: float, on_text=None) -> dict:
    """One streamed request. Adds 'reasoned' (did the model think?) and 'provider' to the result."""
    # OpenRouter-specific options go in "extra_body" (they are not part of OpenAI's own API).
    # Reasoning is always switched on; with require_parameters, only hosts that support it are used.
    provider_opts = {'require_parameters': True}
    if excluded:
        provider_opts['ignore'] = list(excluded)
    reasoning = {'enabled': True}
    if effort != 'default':
        reasoning['effort'] = effort
    extra = {'provider': provider_opts, 'reasoning': reasoning}   # the cost always comes with the usage

    try:
        client = openai.OpenAI(base_url=BASE_URL, api_key=key, timeout=1800)
        stream = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            response_format={'type': 'json_schema',
                             'json_schema': {'name': 'genotype_answers', 'strict': True, 'schema': ANSWER_SCHEMA}},
            temperature=temperature,
            seed=seed,
            stream=True,
            stream_options={'include_usage': True},   # token counts arrive in the last chunk
            extra_body=extra,
        )
        # Collect the streamed answer piece by piece.
        parts, finish, served, usage, host, reasoning_chars, gen_id, cost = [], None, None, {}, None, 0, None, None
        for chunk in stream:
            gen_id = gen_id or getattr(chunk, 'id', None)
            served = served or getattr(chunk, 'model', None)
            provider = getattr(chunk, 'provider', None) or (chunk.model_extra or {}).get('provider')
            if provider:
                host = provider
                served = f'{chunk.model} via {provider}'
            if chunk.usage:
                details = getattr(chunk.usage, 'completion_tokens_details', None)
                usage = dict(input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens,
                             reasoning_tokens=getattr(details, 'reasoning_tokens', None) if details else None)
                reported = getattr(chunk.usage, 'cost', None)
                if reported is None:
                    reported = (chunk.usage.model_extra or {}).get('cost')
                cost = float(reported) if reported is not None else cost
            for choice in chunk.choices or []:
                if choice.delta and choice.delta.content:
                    parts.append(choice.delta.content)
                    if on_text:
                        on_text(choice.delta.content)
                if choice.delta:                     # the thinking arrives as "reasoning" (not saved)
                    thought = (choice.delta.model_extra or {}).get('reasoning')
                    if thought:
                        reasoning_chars += len(thought)
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
        note = 'The reply was cut off at the host\'s maximum output length (thinking counts too); ' \
               'try sending one request per family.'
    elif stop == 'refusal':
        note = 'The provider filtered the reply.'
    elif finish is None:
        note = 'The stream ended without a finish reason; the reply may be incomplete.'
    reasoned = bool(reasoning_chars) or bool(usage.get('reasoning_tokens'))
    seconds = round(time.time() - started, 1)     # measured before the cost lookup below
    if cost is None and gen_id:
        cost = _generation_cost(gen_id, key)
    return dict(text=''.join(parts), stop_reason=stop or 'unknown', served_by=served or model, usage=usage,
                seconds=seconds, error=note, reasoned=reasoned, provider=host, cost_usd=cost,
                cost_source='reported by OpenRouter' if cost is not None else None, request_id=gen_id)


def _generation_cost(gen_id: str, key: str, tries: int = 3) -> Optional[float]:
    """The cost OpenRouter recorded for one request (its stats can take a moment to appear)."""
    for k in range(tries):
        try:
            req = urllib.request.Request(f'{BASE_URL}/generation?id={gen_id}',
                                         headers={'Authorization': f'Bearer {key}'})
            with urllib.request.urlopen(req, timeout=20) as r:
                total = (json.loads(r.read()).get('data') or {}).get('total_cost')
            if total is not None:
                return float(total)
        except Exception:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def _failed(started: float, message: str) -> dict:
    return dict(text='', stop_reason=None, served_by=None, usage={}, seconds=round(time.time() - started, 1),
                error=message, cost_usd=None, cost_source=None)
