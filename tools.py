"""Wiederverwendbare LangChain Tools für Web-Suche, Wetter, Zeit und Rechner."""
import requests
import ast
import operator
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
    """Gibt das aktuelle Wetter für eine Stadt zurück (Open-Meteo API)."""
    try:
        # Step 1: Geocoding - Convert city name to coordinates using Open-Meteo's geocoding API
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_resp = requests.get(geocode_url, timeout=10)

        if geo_resp.status_code != 200:
            return f"Stadt nicht gefunden: {city}"

        geo_data = geo_resp.json()
        if not geo_data.get("results"):
            return f"Stadt nicht gefunden: {city}"

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        location_name = location.get("name", city)
        country = location.get("country", "")

        # Step 2: Get current weather from Open-Meteo
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            f"&timezone=auto"
        )
        weather_resp = requests.get(weather_url, timeout=10)

        if weather_resp.status_code != 200:
            return f"Wetterdaten nicht verfügbar für {city}"

        weather_data = weather_resp.json()
        current = weather_data.get("current", {})

        # Weather code mapping (WMO codes)
        weather_codes = {
            0: "Klar", 1: "Überwiegend klar", 2: "Teilweise bewölkt", 3: "Bewölkt",
            45: "Neblig", 48: "Neblig mit Reifablagerung",
            51: "Leichter Nieselregen", 53: "Mäßiger Nieselregen", 55: "Starker Nieselregen",
            61: "Leichter Regen", 63: "Mäßiger Regen", 65: "Starker Regen",
            71: "Leichter Schneefall", 73: "Mäßiger Schneefall", 75: "Starker Schneefall",
            80: "Leichte Regenschauer", 81: "Mäßige Regenschauer", 82: "Heftige Regenschauer",
            95: "Gewitter", 96: "Gewitter mit leichtem Hagel", 99: "Gewitter mit starkem Hagel"
        }

        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind_speed = current.get("wind_speed_10m", "N/A")
        weather_code = current.get("weather_code", 0)
        condition = weather_codes.get(weather_code, "Unbekannt")

        result = (
            f"{location_name}, {country}: {condition}, {temp}°C, "
            f"Luftfeuchtigkeit {humidity}%, Wind {wind_speed} km/h"
        )
        return result

    except requests.exceptions.Timeout:
        return f"Timeout beim Abrufen der Wetterdaten für {city}"
    except requests.exceptions.RequestException as e:
        return f"Netzwerkfehler: {e}"
    except Exception as e:
        return f"Wetterfehler: {e}"


def time_tool(dummy: str = "") -> str:
    """Gibt das aktuelle Datum und Uhrzeit zurück."""
    now = datetime.now()
    return f"Сегодня {now.strftime('%d.%m.%Y')}, время {now.strftime('%H:%M:%S')}"


def calc_tool(expr: str) -> str:
    """Berechnet mathematische Ausdrücke sicher via AST-Parsing."""
    # Define safe operations
    safe_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval_node(node):
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_type = type(node.op)
            if op_type not in safe_operators:
                raise ValueError(f"Unsupported operation: {op_type.__name__}")
            return safe_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in safe_operators:
                raise ValueError(f"Unsupported operation: {op_type.__name__}")
            return safe_operators[op_type](operand)
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    try:
        # Parse the expression into an AST
        tree = ast.parse(expr, mode='eval')
        # Evaluate the AST safely
        result = _eval_node(tree.body)
        return str(result)
    except SyntaxError:
        return "Fehler: Ungültige mathematische Syntax"
    except ZeroDivisionError:
        return "Fehler: Division durch Null"
    except ValueError as e:
        return f"Fehler: {str(e)}"
    except Exception as e:
        return f"Rechenfehler: {str(e)}"


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