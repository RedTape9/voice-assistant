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