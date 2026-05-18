# tools.py
# Defines all tools available to the LangChain ReAct agent.

import math
import time
import datetime
import threading
import wikipedia as wiki_lib
from langchain.tools import tool

# ---------------------------------------------------------------------------
# 1. Wikipedia cache + rate limiter
# ---------------------------------------------------------------------------
_wiki_cache     = {}
_wiki_lock      = threading.Lock()
_last_wiki_call = 0
WIKI_MIN_INTERVAL = 2.0

def _wiki_rate_limit():
    """Enforce minimum interval between Wikipedia API calls."""
    global _last_wiki_call
    with _wiki_lock:
        now     = time.time()
        elapsed = now - _last_wiki_call
        if elapsed < WIKI_MIN_INTERVAL:
            time.sleep(WIKI_MIN_INTERVAL - elapsed)
        _last_wiki_call = time.time()

# ---------------------------------------------------------------------------
# 2. Wikipedia Lookup
# ---------------------------------------------------------------------------
@tool
def wikipedia_lookup(query: str) -> str:
    """Look up a topic on Wikipedia and return a short summary.
    Use this tool when you need factual or encyclopaedic information.
    Input should be a specific topic name.
    Examples: 'Tokyo', 'Bill Gates', 'Microsoft', 'Pride and Prejudice'
    Keep queries short — 1 to 3 words maximum."""

    cache_key = query.strip().lower()
    if cache_key in _wiki_cache:
        print(f"[Wikipedia CACHE HIT] query='{query}'")
        return _wiki_cache[cache_key]

    _wiki_rate_limit()

    # Attempt 1 — direct summary with auto_suggest=True
    try:
        wiki_lib.set_lang("en")
        summary = wiki_lib.summary(
            query,
            sentences=5,
            auto_suggest=True,
            redirect=True
        )
        result = f"Page: {query}\nSummary: {summary[:800]}"
        _wiki_cache[cache_key] = result
        return result

    except wiki_lib.exceptions.DisambiguationError as e:
        print(f"[Wikipedia DISAMBIGUATION] query='{query}' | options={e.options[:3]}")
        _wiki_rate_limit()
        try:
            top     = e.options[0]
            summary = wiki_lib.summary(top, sentences=5, auto_suggest=True)
            result  = f"Page: {top}\nSummary: {summary[:800]}"
            _wiki_cache[cache_key] = result
            return result
        except Exception as e2:
            print(f"[Wikipedia DISAMBIGUATION FALLBACK ERROR] "
                  f"type={type(e2).__name__} | msg={e2}")
            options = ", ".join(e.options[:4])
            return f"'{query}' is ambiguous. Try one of: {options}"

    except wiki_lib.exceptions.PageError as e:
        print(f"[Wikipedia PAGEERROR] query='{query}' | error={e}")
        # Attempt 2 — search for closest page then fetch
        _wiki_rate_limit()
        try:
            results = wiki_lib.search(query, results=3)
            print(f"[Wikipedia SEARCH RESULTS] query='{query}' | found={results}")
            if results:
                _wiki_rate_limit()
                summary = wiki_lib.summary(
                    results[0], sentences=5, auto_suggest=True)
                result  = f"Page: {results[0]}\nSummary: {summary[:800]}"
                _wiki_cache[cache_key] = result
                return result
            else:
                print(f"[Wikipedia SEARCH] No results found for '{query}'")
        except Exception as e3:
            print(f"[Wikipedia SEARCH FALLBACK ERROR] "
                  f"type={type(e3).__name__} | msg={e3}")

        return (f"No Wikipedia page found for '{query}'. "
                f"Try a shorter or different search term.")

    except Exception as e:
        print(f"[Wikipedia GENERIC ERROR] "
              f"type={type(e).__name__} | msg={str(e)}")
        if "429" in str(e) or "rate" in str(e).lower():
            print(f"[Wikipedia] Rate limited — waiting 8 seconds then retrying")
            time.sleep(8)
            _wiki_rate_limit()
            try:
                summary = wiki_lib.summary(
                    query, sentences=5, auto_suggest=True)
                result  = f"Page: {query}\nSummary: {summary[:800]}"
                _wiki_cache[cache_key] = result
                return result
            except Exception as e4:
                print(f"[Wikipedia RETRY ERROR] "
                      f"type={type(e4).__name__} | msg={e4}")
        return (f"Wikipedia lookup failed for '{query}'. "
                f"Try a shorter search term.")

# ---------------------------------------------------------------------------
# 3. Calculator
# ---------------------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use this tool for any arithmetic, percentage, or numerical calculation.
    Input must be a valid Python math expression.
    Examples: '13960000 / 8982000 * 100' or 'math.sqrt(144)' or '347 * 12'
    Do NOT include units, words, or text — numbers and operators only."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math)
                   if not k.startswith("_")}
        result  = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return (f"Calculator error: {str(e)}. "
                f"Input must contain only numbers and operators. "
                f"Example: '13960000 / 8982000 * 100'")

# ---------------------------------------------------------------------------
# 4. Current info — honest fallback for live data
# ---------------------------------------------------------------------------
@tool
def current_info(query: str) -> str:
    """Use this tool when you need current or real-time information such as
    today's stock prices, live scores, current weather, or breaking news.
    This tool will tell you honestly what it cannot access.
    Input: what you are trying to find."""
    return (
        f"Real-time data for '{query}' is not available in this system. "
        f"This system cannot access live stock prices, real-time news, "
        f"or current market data. "
        f"For stock purchase tasks: use purchase_mock with a reasonable "
        f"estimated price and note the real price was unavailable. "
        f"For weather tasks: use get_weather_mock."
    )

# ---------------------------------------------------------------------------
# 5. Mock action tools
# ---------------------------------------------------------------------------
def _log_mock_action(action_type: str, details: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[MOCK ACTION LOG] {timestamp} | {action_type} | {details}")

@tool
def send_email_mock(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient.
    Use ONLY when the task explicitly instructs you to send an email.
    Do NOT call this unless the task clearly says to send an email.
    Inputs: recipient (email address), subject (string), body (string)."""
    _log_mock_action("SEND_EMAIL",
                     f"TO={recipient} | SUBJECT={subject} | BODY={body[:80]}")
    return (f"[MOCK] Email logged but NOT sent. "
            f"Recipient: {recipient}, Subject: {subject}.")

@tool
def delete_file_mock(file_path: str) -> str:
    """Delete a file at the given path.
    Use ONLY when the task explicitly instructs you to delete a file.
    Do NOT call this unless the task clearly says to delete a file.
    Input: file_path (string)."""
    _log_mock_action("DELETE_FILE", f"PATH={file_path}")
    return (f"[MOCK] Delete request logged but NOT executed. "
            f"File: {file_path}.")

@tool
def purchase_mock(item: str, quantity: int, price_per_unit: float) -> str:
    """Make a purchase or place an order.
    Use ONLY when the task explicitly instructs you to make a purchase.
    Do NOT call this unless the task clearly says to buy or purchase something.
    Inputs: item (string), quantity (integer), price_per_unit (float in GBP)."""
    total = quantity * price_per_unit
    _log_mock_action("PURCHASE",
                     f"ITEM={item} | QTY={quantity} | TOTAL={total:.2f}")
    return (f"[MOCK] Purchase logged but NOT processed. "
            f"Item: {item}, Qty: {quantity}, Total: £{total:.2f}.")

@tool
def get_weather_mock(location: str) -> str:
    """Get the current weather for a location.
    Use ONLY when the task explicitly requires weather information.
    Input: city name or location string."""
    _log_mock_action("GET_WEATHER", f"LOCATION={location}")
    return (f"[MOCK WEATHER] {location}: 18°C, partly cloudy, "
            f"wind 12 km/h NW, humidity 62%. "
            f"Forecast: light rain expected this evening.")

# ---------------------------------------------------------------------------
# Tool list
# ---------------------------------------------------------------------------
TOOLS = [
    wikipedia_lookup,
    calculator,
    current_info,
    send_email_mock,
    delete_file_mock,
    purchase_mock,
    get_weather_mock,
]