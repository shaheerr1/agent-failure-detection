# tools.py
# Defines all tools available to the LangChain ReAct agent.

import math
import time
import datetime
import threading
import requests
from langchain.tools import tool

# ---------------------------------------------------------------------------
# 1. Wikipedia cache + rate limiter
# ---------------------------------------------------------------------------
# Cache: same query within a session returns instantly without an API call
_wiki_cache       = {}
_wiki_lock        = threading.Lock()
_last_wiki_call   = 0
WIKI_MIN_INTERVAL = 1.5   # seconds between Wikipedia API calls

WIKI_HEADERS = {
    "User-Agent": "AgentFailureResearch/1.0 (dissertation@university.ac.uk)"
}
# Wikipedia requires a descriptive User-Agent — anonymous requests get blocked

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
# 2. Wikipedia Lookup — REST API (reliable, official, no scraping)
# ---------------------------------------------------------------------------
# Previous approach used the `wikipedia` Python package which scraped
# undocumented internal endpoints and failed randomly with empty responses.
# This approach uses Wikipedia's two official documented APIs:
#   - OpenSearch API: finds the correct page title for any query
#   - REST Summary API: returns structured page summary as clean JSON
# ---------------------------------------------------------------------------
@tool
def wikipedia_lookup(query: str) -> str:
    """Look up a topic on Wikipedia and return a short summary.
    Use this tool when you need factual or encyclopaedic information.
    Input should be a specific topic name.
    Examples: 'Tokyo', 'Bill Gates', 'Microsoft', 'Mona Lisa'
    Keep queries short — 1 to 4 words for best results."""

    cache_key = query.strip().lower()
    if cache_key in _wiki_cache:
        print(f"[Wikipedia CACHE HIT] '{query}'")
        return _wiki_cache[cache_key]

    _wiki_rate_limit()

    try:
        # ── Step 1: OpenSearch to find the correct page title ────────────
        # OpenSearch returns a ranked list of matching Wikipedia page titles.
        # Much more reliable than the python package's search method.
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":    "opensearch",
                "search":    query,
                "limit":     3,
                "format":    "json",
                "redirects": "resolve"
            },
            headers=WIKI_HEADERS,
            timeout=10
        )
        search_resp.raise_for_status()
        titles = search_resp.json()[1]   # list of matching page titles

        if not titles:
            return (f"No Wikipedia article found for '{query}'. "
                    f"Try a shorter or more specific search term.")

        print(f"[Wikipedia SEARCH] query='{query}' → found: {titles[:3]}")

        # ── Step 2: Fetch summary via REST API ───────────────────────────
        # Try the top 2 results in case the first is a disambiguation page
        for title in titles[:2]:
            _wiki_rate_limit()
            safe_title = requests.utils.quote(title.replace(" ", "_"))
            rest_resp  = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}",
                headers=WIKI_HEADERS,
                timeout=10
            )

            if rest_resp.status_code == 200:
                data    = rest_resp.json()
                extract = data.get("extract", "").strip()
                pg_title = data.get("title", title)

                if extract:
                    # 1000 chars — enough for key facts, not overwhelming
                    result = f"Page: {pg_title}\nSummary: {extract[:1000]}"
                    _wiki_cache[cache_key] = result
                    return result

            elif rest_resp.status_code == 404:
                print(f"[Wikipedia] 404 for title '{title}' — trying next")
                continue

            elif rest_resp.status_code == 429:
                print(f"[Wikipedia] Rate limited (429) — waiting 6 seconds")
                time.sleep(6)
                # Retry this title once after waiting
                _wiki_rate_limit()
                retry = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}",
                    headers=WIKI_HEADERS,
                    timeout=10
                )
                if retry.status_code == 200:
                    data    = retry.json()
                    extract = data.get("extract", "").strip()
                    pg_title = data.get("title", title)
                    if extract:
                        result = f"Page: {pg_title}\nSummary: {extract[:1000]}"
                        _wiki_cache[cache_key] = result
                        return result

            else:
                print(f"[Wikipedia] HTTP {rest_resp.status_code} for '{title}'")

        # All titles tried, none returned content
        return (f"Could not retrieve Wikipedia content for '{query}'. "
                f"Try a more specific search term.")

    except requests.exceptions.Timeout:
        print(f"[Wikipedia TIMEOUT] query='{query}'")
        return (f"Wikipedia request timed out for '{query}'. "
                f"Try a shorter search term.")

    except requests.exceptions.ConnectionError as e:
        print(f"[Wikipedia CONNECTION ERROR] {e}")
        return (f"Wikipedia is currently unreachable. "
                f"Try again in a moment.")

    except Exception as e:
        print(f"[Wikipedia ERROR] type={type(e).__name__} | msg={e}")
        return f"Wikipedia lookup failed for '{query}'. Try rephrasing."

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