"""Wiederverwendbare LangChain Tools für Web-Suche, Wetter, Zeit und Rechner."""
import requests
from datetime import datetime
from langchain.tools import Tool


def web_search_tool(query: str) -> str:
    """Sucht im Internet nach aktuellen Informationen (News, Fakten, etc.)."""
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=3)
        if not results:
            return "Keine Ergebnisse gefunden."
        snippets = [f"• {r['title']}\n  {r['body'][:200]}..." for r in results]
        return "\n\n".join(snippets)
    except Exception as e:
        return f"Suchfehler: {e}"


def weather_tool(city: str) -> str:
    """Gibt das aktuelle Wetter für eine Stadt zurück (wttr.in API)."""
    try:
        url = f"https://wttr.in/{city}?format=%l:+%C+%t,+влажность+%h,+ветер+%w&lang=ru"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return f"Wetter nicht verfügbar für {city}"
        return resp.text.strip()
    except Exception as e:
        return f"Wetterfehler: {e}"


def time_tool(dummy: str = "") -> str:
    """Gibt das aktuelle Datum und Uhrzeit zurück."""
    now = datetime.now()
    return f"Сегодня {now.strftime('%d.%m.%Y')}, время {now.strftime('%H:%M:%S')}"


def calc_tool(expr: str) -> str:
    """Berechnet mathematische Ausdrücke (sicher via restricted eval)."""
    try:
        # Sicherheitscheck: nur Zahlen, Operatoren, Klammern erlaubt
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expr):
            return "Ungültiger Ausdruck (nur Zahlen und Operatoren erlaubt)"
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Rechenfehler: {e}"


# Vordefinierte Tool-Liste (exportiert)
DEFAULT_TOOLS = [
    Tool(
        name="web_search",
        func=web_search_tool,
        description="Sucht im Internet nach aktuellen Informationen (News, Fakten, Erklärungen). Nutze IMMER das Jahr 2025 in der Suche. Input: Suchanfrage als Text."
    ),
    Tool(
        name="weather",
        func=weather_tool,
        description="Gibt das aktuelle Wetter für eine Stadt zurück. Input: Stadtname auf Englisch oder Russisch (z.B. 'Hamburg', 'Москва')."
    ),
    Tool(
        name="current_time",
        func=time_tool,
        description="Gibt das aktuelle Datum und Uhrzeit zurück. Nutze dieses Tool bei Fragen nach Datum, Uhrzeit oder 'heute'. Input: leerer String."
    ),
    Tool(
        name="calculator",
        func=calc_tool,
        description="Berechnet mathematische Ausdrücke wie '2+2' oder '12*8'. Input: mathematischer Ausdruck als String."
    )
]