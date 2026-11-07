"""
llm.py — asking Claude through the Anthropic API
================================================

One function, `ask_claude()`, sends the prompt from prompt.py and returns the reply text plus a
record of what happened (model, stop reason, token usage). Things to know:

* NO TOOLS are given to the model: no web search, no code execution, no files. The prompt is all it
  has (part of the no-leakage guarantee) and it must reason, not run a solver.
* STRUCTURED OUTPUT: the answer format (prompt.ANSWER_SCHEMA) is passed as a JSON Schema, so the
  reply is always valid JSON in the requested shape.
* STREAMING is used because a reply can be long; it avoids HTTP time-outs.
* THINKING: current Claude models always think (adaptive thinking); how hard is set by "effort"
  (low / medium / high / xhigh / max). This is recorded with every run.
* DETERMINISM: current Claude models do not accept a temperature setting, so two calls with the same
  prompt can give different answers. What IS deterministic is the scoring: the exact reply is saved
  with each run, and re-scoring it always gives the same numbers.
* FALLBACK MODELS are OFF by default. If a model declines a request, the API can re-run it on
  another model — but then the answer would be credited to the wrong model. Leave it off for a fair
  evaluation; if you switch it on, the model that actually answered is recorded.
* CREDENTIALS come from the ANTHROPIC_API_KEY environment variable (or an `ant auth login`
  profile), or from the key typed into the app. Keys are never written to disk.
"""

import time
from typing import Optional

import anthropic

from prompt import ANSWER_SCHEMA

# Models offered in the app (newest first). Any other model ID can also be typed in.
MODELS = ['claude-opus-5-5', 'claude-fable-5-1', 'claude-sonnet-5-5', 'claude-haiku-5-5']
EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max']
MAX_TOKENS = 64000          # room for long thinking + a long JSON answer


def ask_claude(system: str, user: str, model: str, effort: str, api_key: Optional[str] = None,
               allow_fallback: bool = False) -> dict:
    """Send one request. Returns a dict with:
         text           the reply (JSON text), '' if there was none
         stop_reason    'end_turn' when finished normally; 'max_tokens' = cut off; 'refusal' = declined
         served_by      the model that actually produced the answer
         usage          input/output token counts
         seconds        how long it took
         error          an error message, or None"""
    request = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{'role': 'user', 'content': user}],
        output_config={'effort': effort,
                       'format': {'type': 'json_schema', 'schema': ANSWER_SCHEMA}},
    )
    started = time.time()
    try:
        # Created inside the try: with no key configured at all, this raises an AnthropicError
        # that is reported in the app instead of crashing it.
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        if allow_fallback:
            # Beta endpoint: lets the API hand a declined request to Anthropic's recommended
            # fallback model. Off by default (see the note at the top of this file).
            with client.beta.messages.stream(betas=['server-side-fallback-2026-07-01'],
                                             fallbacks='default', **request) as stream:
                msg = stream.get_final_message()
        else:
            with client.messages.stream(**request) as stream:
                msg = stream.get_final_message()
    # Most specific errors first, so the message shown in the app says what to do.
    except anthropic.AuthenticationError:
        return _failed(started, 'The API key was rejected. Check ANTHROPIC_API_KEY or the key you typed.')
    except anthropic.NotFoundError:
        return _failed(started, f'Model "{model}" was not found for this API key.')
    except anthropic.RateLimitError:
        return _failed(started, 'Rate limit reached. Wait a minute and try again.')
    except anthropic.BadRequestError as e:
        return _failed(started, f'The API refused the request: {e.message}')
    except anthropic.APIStatusError as e:
        return _failed(started, f'API error {e.status_code}: {e.message}')
    except anthropic.APIConnectionError:
        return _failed(started, 'Could not reach the Anthropic API. Check the internet connection.')
    except anthropic.AnthropicError as e:
        return _failed(started, str(e))
    except TypeError as e:
        # The SDK raises TypeError("Could not resolve authentication method…") when no key, token
        # or profile is configured at all.
        if 'authentication' not in str(e):
            raise
        return _failed(started, 'No API key found. Type one in, or set ANTHROPIC_API_KEY before '
                                'starting the app.')

    # The answer is in the text block(s); thinking blocks are skipped.
    text = ''.join(b.text for b in msg.content if b.type == 'text')
    usage = dict(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
    note = None
    if msg.stop_reason == 'refusal':
        note = 'The model declined to answer.'
    elif msg.stop_reason == 'max_tokens':
        note = 'The reply was cut off at the token limit; it may be incomplete or unreadable.'
    return dict(text=text, stop_reason=msg.stop_reason, served_by=msg.model, usage=usage,
                seconds=round(time.time() - started, 1), error=note)


def _failed(started: float, message: str) -> dict:
    return dict(text='', stop_reason=None, served_by=None, usage={}, seconds=round(time.time() - started, 1),
                error=message)
