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
* COST is estimated from the token counts and the per-token prices in PRICES (Anthropic's
  first-party API list prices, as of the date in PRICES_AS_OF). Thinking tokens are billed as output
  and are included in output_tokens. A model not in the table gets no cost (shown as "—").
* CREDENTIALS come from the ANTHROPIC_API_KEY environment variable (or an `ant auth login`
  profile), or from the key typed into the app. Keys are never written to disk.
"""

import time
from typing import Callable, Optional

import anthropic

from prompt import ANSWER_SCHEMA

# Models offered in the app (newest first). Any other model ID can also be typed in.
MODELS = ['claude-opus-5-5', 'claude-fable-5-1', 'claude-sonnet-5-5', 'claude-haiku-5-5']
EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max']
MAX_TOKENS = 64000          # room for long thinking + a long JSON answer

# US dollars per million tokens: input, output, cache read, cache write (5-minute cache = 1.25 x input).
# Claude Haiku 5.5 costs more for prompts over 100K tokens (the second entry).
PRICES_AS_OF = '2026-10-06'
PRICES = {
    'claude-opus-5-5':   dict(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    'claude-fable-5-1':  dict(input=10.00, output=50.00, cache_read=0.25, cache_write=12.50),
    'claude-sonnet-5-5': dict(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    'claude-haiku-5-5':  dict(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
}
HAIKU_LONG_PROMPT = dict(threshold=100_000, input=0.50, output=2.50, cache_read=0.05, cache_write=0.625)


def estimate_cost(model: str, usage: dict) -> Optional[float]:
    """Dollar cost of one request from its token counts, or None for a model not in PRICES."""
    p = PRICES.get(model)
    if p is None:
        return None
    prompt = sum(usage.get(k) or 0 for k in ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'))
    if model == 'claude-haiku-5-5' and prompt > HAIKU_LONG_PROMPT['threshold']:
        p = HAIKU_LONG_PROMPT
    return round(((usage.get('input_tokens') or 0) * p['input'] + (usage.get('output_tokens') or 0) * p['output']
                  + (usage.get('cache_read_input_tokens') or 0) * p['cache_read']
                  + (usage.get('cache_creation_input_tokens') or 0) * p['cache_write']) / 1e6, 6)


def ask_claude(system: str, user: str, model: str, effort: str, api_key: Optional[str] = None,
               allow_fallback: bool = False, on_text: Optional[Callable[[str], None]] = None) -> dict:
    """Send one request. Returns a dict with:
         text           the reply (JSON text), '' if there was none
         stop_reason    'end_turn' when finished normally; 'max_tokens' = cut off; 'refusal' = declined
         served_by      the model that actually produced the answer
         usage          input/output token counts
         seconds        how long it took
         error          an error message, or None
    `on_text`, if given, is called with each new piece of the answer as it streams in (the app uses
    it to show live how many variants have been answered)."""
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
                msg = _consume(stream, on_text)
        else:
            with client.messages.stream(**request) as stream:
                msg = _consume(stream, on_text)
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
    usage = dict(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                 cache_read_input_tokens=getattr(msg.usage, 'cache_read_input_tokens', None) or 0,
                 cache_creation_input_tokens=getattr(msg.usage, 'cache_creation_input_tokens', None) or 0)
    cost = estimate_cost(msg.model, usage)
    note = None
    if msg.stop_reason == 'refusal':
        note = 'The model declined to answer.'
    elif msg.stop_reason == 'max_tokens':
        note = 'The reply was cut off at the token limit; it may be incomplete or unreadable.'
    return dict(text=text, stop_reason=msg.stop_reason, served_by=msg.model, usage=usage,
                seconds=round(time.time() - started, 1), error=note, cost_usd=cost,
                cost_source=f'estimated from list prices ({PRICES_AS_OF})' if cost is not None else None,
                request_id=getattr(msg, '_request_id', None))


def _consume(stream, on_text):
    """Read the stream to the end, passing each piece of answer text to on_text."""
    if on_text:
        for piece in stream.text_stream:
            on_text(piece)
    return stream.get_final_message()


def _failed(started: float, message: str) -> dict:
    return dict(text='', stop_reason=None, served_by=None, usage={}, seconds=round(time.time() - started, 1),
                error=message, cost_usd=None, cost_source=None)
