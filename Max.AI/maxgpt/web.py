"""Live knowledge for Max from Wikipedia (and its sister sites), with no API key needed.

* search / summaries of articles in any language edition of Wikipedia
* ranking tables from list articles ("top 10 fastest cars")
* word definitions from Wiktionary
* the day's world news from Wikipedia's Current events portal

Everything uses only the Python standard library, a short timeout and an in-memory cache.
"""
import html
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

USER_AGENT = "MaxAI/1.0 (https://github.com/pranovs2012-source/Max-AI; educational chatbot)"
TIMEOUT = 7
_cache = {}


class WebError(Exception):
    pass


def _get(url, params=None, ttl=900):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # network down, blocked, timeout, bad JSON
        raise WebError(str(e)) from e
    if len(_cache) > 500:
        _cache.clear()
    _cache[url] = (time.time(), data)
    return data


def _api(lang):
    return f"https://{lang}.wikipedia.org/w/api.php"


def page_url(title, lang="en"):
    return f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def strip_tags(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


# ── Languages ────────────────────────────────────────────────────────────────
_SCRIPTS = [  # (first, last code point, Wikipedia language)
    (0x0900, 0x097F, "hi"), (0x0980, 0x09FF, "bn"), (0x0A00, 0x0A7F, "pa"), (0x0A80, 0x0AFF, "gu"),
    (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"), (0x0D00, 0x0D7F, "ml"),
    (0x0600, 0x06FF, "ar"), (0x0400, 0x04FF, "ru"), (0x3040, 0x30FF, "ja"), (0xAC00, 0xD7AF, "ko"),
    (0x4E00, 0x9FFF, "zh"), (0x0E00, 0x0E7F, "th"), (0x0590, 0x05FF, "he"), (0x0370, 0x03FF, "el"),
]


def detect_language(text):
    """Pick the Wikipedia edition for the question's script (English for Latin text)."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in _SCRIPTS:
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1
    if counts:
        lang, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= 2:
            return lang
    return "en"


# ── Search and articles ──────────────────────────────────────────────────────
def search(query, limit=6, lang="en"):
    data = _get(_api(lang), dict(action="query", list="search", srsearch=query, srlimit=limit,
                                 srprop="snippet", format="json", formatversion=2))
    return [{"title": r["title"], "snippet": strip_tags(r.get("snippet"))}
            for r in data.get("query", {}).get("search", [])]


def articles(titles, lang="en", intro=True):
    """Plain-text extracts (intro only, or the whole article for one title) plus URL and description."""
    params = dict(action="query", prop="extracts|info|description|pageimages", explaintext=1, redirects=1,
                  inprop="url", piprop="thumbnail", pithumbsize=320, titles="|".join(titles[:20]),
                  format="json", formatversion=2)
    if intro:
        params["exintro"] = 1
    data = _get(_api(lang), params)
    query = data.get("query", {})
    # follow normalisation and redirects so results can be matched to the requested titles
    alias = {}
    for key in ("normalized", "redirects"):
        for m in query.get(key, []):
            alias[m["from"]] = m["to"]
    pages = {}
    for p in query.get("pages", []):
        if p.get("missing") or not p.get("extract"):
            continue
        pages[p["title"]] = {
            "title": p["title"],
            "extract": p["extract"].strip(),
            "description": p.get("description", ""),
            "url": p.get("fullurl") or page_url(p["title"], lang),
            "image": (p.get("thumbnail") or {}).get("source"),
        }
    out = []
    for t in titles:
        while t in alias:
            t = alias[t]
        if t in pages and pages[t] not in out:
            out.append(pages[t])
    return out


def article_html(title, lang="en"):
    data = _get(_api(lang), dict(action="parse", page=title, prop="text", redirects=1,
                                 disableeditsection=1, format="json", formatversion=2))
    if "error" in data:
        raise WebError(data["error"].get("info", "parse error"))
    return data["parse"]["title"], data["parse"]["text"]


# ── Tables ───────────────────────────────────────────────────────────────────
class _TableParser(HTMLParser):
    """Collects every `wikitable` as a grid of cell texts, with rowspan/colspan filled in."""
    SKIP_CLASSES = ("reference", "sortkey", "mw-editsection", "noprint", "sr-only", "mw-cite-backlink")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []           # finished tables: {"caption", "rows": [(is_header, [cells])]}
        self.stack = []            # open tables (nested ones are tracked but not collected)
        self.skip = 0              # depth inside hidden / reference elements
        self.skip_tags = []
        self.cell = None
        self.row = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        style = (a.get("style") or "").replace(" ", "")
        if self.skip:
            if tag not in ("br", "img", "hr", "wbr", "meta", "link"):
                self.skip_tags.append(tag)
            return
        if "display:none" in style or any(c in cls.split() for c in self.SKIP_CLASSES) or tag in ("style", "script"):
            self.skip = 1
            self.skip_tags = [tag]
            return
        if tag == "table":
            self.stack.append({"wiki": "wikitable" in cls, "caption": "", "rows": [], "in_caption": False})
        elif not self.stack:
            return
        elif tag == "caption":
            self.stack[-1]["in_caption"] = True
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = {"th": tag == "th", "text": [], "rowspan": _int(a.get("rowspan")),
                         "colspan": _int(a.get("colspan"))}
        elif tag == "br" and self.cell is not None:
            self.cell["text"].append(" ")

    def handle_endtag(self, tag):
        if self.skip:
            if self.skip_tags and self.skip_tags[-1] == tag:
                self.skip_tags.pop()
            elif tag in self.skip_tags:
                while self.skip_tags and self.skip_tags.pop() != tag:
                    pass
            if not self.skip_tags:
                self.skip = 0
            return
        if not self.stack:
            return
        t = self.stack[-1]
        if tag == "caption":
            t["in_caption"] = False
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            text = re.sub(r"\s+", " ", "".join(self.cell["text"])).strip()
            self.row.append((self.cell["th"], text, self.cell["rowspan"], self.cell["colspan"]))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                t["rows"].append(self.row)
            self.row = None
        elif tag == "table":
            done = self.stack.pop()
            if done["wiki"] and done["rows"]:
                self.tables.append({"caption": done["caption"].strip(), "rows": _expand(done["rows"])})
            self.row, self.cell = None, None

    def handle_data(self, data):
        if self.skip or not self.stack:
            return
        if self.cell is not None:
            self.cell["text"].append(data)
        elif self.stack[-1]["in_caption"]:
            self.stack[-1]["caption"] += data


def _int(v):
    try:
        return max(1, min(int(re.sub(r"\D", "", v or "") or 1), 200))
    except ValueError:
        return 1


def _expand(rows):
    """Turn rows of (is_header, text, rowspan, colspan) into a full grid."""
    grid, carry = [], {}  # carry[col] = [remaining rows, is_header, text]
    for row in rows:
        out, col, cells = [], 0, list(row)
        while cells or any(c >= col and carry[c][0] > 0 for c in carry):
            if col in carry and carry[col][0] > 0:
                carry[col][0] -= 1
                out.append((carry[col][1], carry[col][2]))
                col += 1
                continue
            if not cells:
                if col > 60:
                    break
                out.append((False, ""))
                col += 1
                continue
            is_th, text, rs, cs = cells.pop(0)
            for _ in range(cs):
                if rs > 1:
                    carry[col] = [rs - 1, is_th, text]
                out.append((is_th, text))
                col += 1
        grid.append((all(h for h, _ in out), [t for _, t in out]))
    return grid


def tables(page_html):
    p = _TableParser()
    p.feed(page_html)
    result = []
    for t in p.tables:
        header_rows = []
        body = []
        for is_header, cells in t["rows"]:
            if is_header and not body:
                header_rows.append(cells)
            elif not is_header:
                body.append(cells)
        if not body:
            continue
        width = max(len(r) for r in body)
        headers = []
        for c in range(width):
            parts = []
            for hr in header_rows:
                if c < len(hr) and hr[c] and hr[c] not in parts:
                    parts.append(hr[c])
            headers.append(" / ".join(parts))
        result.append({"caption": t["caption"], "headers": headers,
                       "rows": [r + [""] * (width - len(r)) for r in body]})
    return result


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def number(text):
    """First number in a cell ("490.48 km/h (304.77 mph)" -> 490.48), or None."""
    m = _NUM.search((text or "").replace("−", "-"))
    if not m:
        return None
    try:
        value = float(m.group().replace(",", ""))
    except ValueError:
        return None
    tail = text[m.end():m.end() + 12].lower()
    if tail.strip().startswith(("billion", "bn")):
        value *= 1e9
    elif tail.strip().startswith(("million", "mn")):
        value *= 1e6
    elif tail.strip().startswith("trillion"):
        value *= 1e12
    return value


# ── News ─────────────────────────────────────────────────────────────────────
class _ListParser(HTMLParser):
    """Leaf <li> items of a page (news headlines on the Current events portal)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items, self.stack, self.skip = [], [], 0

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if self.skip or "reference" in cls or tag in ("style", "script"):
            self.skip += 1
            return
        if tag == "li":
            self.stack.append({"text": [], "children": 0})
        elif tag in ("ul", "ol") and self.stack:
            self.stack[-1]["children"] += 1

    def handle_endtag(self, tag):
        if self.skip:
            self.skip -= 1
            return
        if tag == "li" and self.stack:
            li = self.stack.pop()
            text = re.sub(r"\s+", " ", "".join(li["text"])).strip()
            if not li["children"] and len(text) > 40:
                self.items.append(text)

    def handle_data(self, data):
        if not self.skip and self.stack:
            for li in self.stack:
                li["text"].append(data)


def news(days=2, limit=8):
    """Recent world news from Wikipedia's Current events portal: [(date, headline)]."""
    out = []
    today = datetime.now(timezone.utc).date()
    for back in range(days):
        day = today - timedelta(days=back)
        title = f"Portal:Current events/{day.year} {day.strftime('%B')} {day.day}"
        try:
            _, page = article_html(title)
        except (WebError, KeyError):
            continue
        p = _ListParser()
        p.feed(page)
        out += [(day.isoformat(), item) for item in p.items]
        if len(out) >= limit:
            break
    return out[:limit]


# ── Dictionary ───────────────────────────────────────────────────────────────
def define(word):
    """Definitions from Wiktionary: [(part of speech, definition)]."""
    url = "https://en.wiktionary.org/api/rest_v1/page/definition/" + urllib.parse.quote(word.strip().replace(" ", "_"))
    try:
        data = _get(url)
    except WebError:
        return []
    out = []
    for entry in data.get("en", []):
        for d in entry.get("definitions", [])[:2]:
            text = strip_tags(d.get("definition"))
            if text:
                out.append((entry.get("partOfSpeech", ""), text))
    return out[:4]


if __name__ == "__main__":
    # python -m maxgpt.web "query"            search and summaries
    # python -m maxgpt.web --tables "Title"   the tables Max can read from an article
    import sys
    if sys.argv[1:2] == ["--tables"]:
        title, page = article_html(" ".join(sys.argv[2:]))
        for i, t in enumerate(tables(page)):
            print(f"[{title} · table {i}] caption={t['caption']!r} rows={len(t['rows'])}")
            print("   headers:", t["headers"])
            for r in t["rows"][:3]:
                print("   ", r)
    else:
        q = " ".join(sys.argv[1:]) or "Python programming language"
        hits = search(q)
        print(hits)
        for a in articles([h["title"] for h in hits[:2]]):
            print(a["title"], "-", a["extract"][:300])
