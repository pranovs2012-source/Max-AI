"""Exact answers for simple arithmetic ("what is 7 * 8?", "12 divided by 4").

Language models are bad at arithmetic, so Max calculates instead of guessing.
Only numbers and + - * / % ** ( ) are allowed: the expression is parsed with
Python's ast module and evaluated by hand, never with eval().
"""
import ast
import operator
import re

_WORDS = [(r"\bmultiplied by\b|\btimes\b|\bx\b", "*"), (r"\bdivided by\b|\bover\b", "/"),
          (r"\bplus\b|\badded to\b", "+"), (r"\bminus\b", "-"), (r"\bto the power of\b|\^", "**"),
          (r"\bmod(ulo)?\b", "%")]
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_PERCENT = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)$")
_PREFIX = re.compile(r"^\s*(what\s+is|what's|whats|calculate|compute|solve|how\s+much\s+is)\s+", re.I)


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("not arithmetic")


def _number(value):
    if isinstance(value, float):
        return int(value) if value.is_integer() else round(value, 6)
    return value


def answer(question):
    """Return a sentence with the result, or None if this isn't a simple calculation."""
    text = _PREFIX.sub("", question.strip().lower()).rstrip("?=. ")
    percent = _PERCENT.match(text)
    if percent:  # "15% of 200"
        share, total = float(percent.group(1)), float(percent.group(2))
        return f"{percent.group(1)}% of {percent.group(2)} = {_number(share * total / 100)}"
    for pattern, symbol in _WORDS:
        text = re.sub(pattern, symbol, text)
    if not re.fullmatch(r"[\d\s.+\-*/%()]+", text) or not re.search(r"\d\s*[-+*/%]", text):
        return None
    try:
        value = _eval(ast.parse(text, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    pretty = " ".join(text.split())
    return f"{pretty} = {_number(value)}"
