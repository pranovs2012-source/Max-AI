"""Max's reasoning loop: understand the question, pick a strategy, gather facts, check, answer.

Every answer comes with the steps Max took ("thinking") and the sources it used, so the
chat can show them like a search engine would. Strategies:

  calculate   exact arithmetic
  recall      MaxGPT's own trained knowledge (identity, coding, everyday topics, world facts),
              with a self-check that regenerates garbled answers
  rank        "top 10 fastest cars": reads ranking tables in Wikipedia list articles
  news        today's world news from Wikipedia's Current events portal
  define      word meanings from Wiktionary
  look up     anything else: searches Wikipedia (in the question's language), reads the best
              articles and picks the sentences that answer the question
"""
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import calc, web

STOP = set("""a an the is are was were be been being to of in on for and or with by at from as it its this that these
those what which who whom whose when where why how do does did can could would should will shall may might
please tell me about give show list top best most some any i you your we our they their he she his her him them
there here know want find explain describe info information details name names much many""".split())
PRONOUNS = {"it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "there", "that", "this", "those"}
ABOUT_MAX = re.compile(r"\b(you|your|yourself|max|maxgpt|pranov)\b", re.I)
CODE_WORDS = re.compile(r"\b(code|python|javascript|js|java|html|css|sql|function|loop|array|list in|dictionary|class|"
                        r"git|bug|error|debug|compile|regex|api|flask|react|node|variable|string|recursion|algorithm)\b", re.I)
SUPERLATIVE = re.compile(r"\b(top\s*\d+|top|best|largest|biggest|fastest|tallest|highest|longest|richest|most|"
                         r"oldest|smallest|slowest|shortest|deepest|heaviest|cheapest|expensive|popular|"
                         r"best[- ]selling|highest[- ]grossing|list of|ranking|ranked)\b", re.I)
NEWS = re.compile(r"\b(news|headlines|current events|what('s| is) happening|today in the world|latest events)\b", re.I)
DEFINE = re.compile(r"^(?:define|definition of|meaning of|what does)\s+(?:the word\s+)?[\"']?([a-z][a-z\- ]{1,40}?)[\"']?"
                    r"(?:\s+mean)?\s*\??$", re.I)

# Which table columns measure each superlative, and which way to sort.
MEASURES = [
    (r"fast|speed|quick", ["top speed", "speed", "km/h", "mph", "velocity"], True),
    (r"slow", ["speed", "km/h", "mph"], False),
    (r"tall|height", ["height", "metres", "meters", "feet", "ft", "floors"], True),
    (r"high|elevation", ["elevation", "height", "altitude", "metres", "m"], True),
    (r"populous|population|people", ["population"], True),
    (r"rich|wealth|net worth|billionaire", ["net worth", "worth", "wealth", "billion", "billionaire"], True),
    (r"long|length", ["length", "km", "mi"], True),
    (r"deep|depth", ["depth"], True),
    (r"heav|mass|weight", ["mass", "weight", "kg", "tonnes"], True),
    (r"expensive|price|cost", ["price", "cost", "us$", "usd"], True),
    (r"cheap", ["price", "cost"], False),
    (r"gross|box office", ["gross", "worldwide", "box office"], True),
    (r"selling|sold|sales|popular", ["sales", "sold", "units", "copies", "certified", "subscribers", "views", "followers"], True),
    (r"large|big|size", ["area", "population", "size", "km2", "capacity", "revenue"], True),
    (r"small|tiny", ["area", "size", "population"], False),
    (r"short", ["length", "height"], False),
    (r"old", ["year", "founded", "established", "date", "born", "age"], False),
    (r"new|recent|latest", ["year", "date", "released"], True),
    (r"valuable|revenue|biggest compan", ["revenue", "market", "value", "valuation"], True),
]
SUPER_WORDS = set("top best most least largest biggest fastest tallest highest longest richest oldest newest smallest "
                  "slowest shortest deepest heaviest cheapest expensive popular populous selling grossing ranked "
                  "ranking famous greatest".split())
NEUTRAL = set("list world global all time ever history record records the by of in and top current".split())
TIME_HEADER = re.compile(r"0\s*[–-]|\(s\)|seconds|\btime\b|lap", re.I)
STARTS_NUMERIC = re.compile(r"^\s*[~≈<>+\-−]?\s*(?:US)?[$€£¥₹]?\s*\d")
NAME_HEADERS = ("name", "model", "car", "vehicle", "country", "city", "company", "person", "title", "building",
                "film", "song", "album", "artist", "player", "team", "mountain", "river", "language", "game", "make")


SMALL_TALK = set("hi hello hey hiya yo sup thank thanks thx bye goodbye ok okay cool nice great good morning "
                 "evening night lol haha wow sure yes no yeah nope fine awesome".split())


def _stem(w):
    """Light stemming so "cities"~"city", "founded"~"found", "created"~"create"."""
    if not w.isascii():
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ous")):
        w = w[:-1]
    if len(w) > 6 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 5 and w.endswith("ed"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def terms(text):
    # split on spaces and punctuation (not \w, which breaks words in scripts like Devanagari)
    words = re.findall(r"[^\s.,!?;:\"'()\[\]{}<>/\\|*~`=_\-\u2013\u2014\u0964]+", text.lower())
    return [_stem(w) for w in words if w not in STOP and len(w) > 1]


class Thinking:
    def __init__(self):
        self.start = time.perf_counter()
        self.steps, self.sources, self.card = [], [], None

    def step(self, title, detail=""):
        self.steps.append({"title": title, "detail": detail,
                           "ms": round((time.perf_counter() - self.start) * 1000)})

    def source(self, title, url):
        if url and all(s["url"] != url for s in self.sources):
            self.sources.append({"title": title, "url": url})

    def result(self, reply, route):
        self.step("Answer ready", route)
        return {"reply": reply.strip(), "thinking": self.steps, "sources": self.sources[:6], "card": self.card,
                "route": route, "seconds": round(time.perf_counter() - self.start, 2)}


class Assistant:
    def __init__(self, engine=None, use_web=True):
        self.engine = engine      # MaxGPTEngine (may be None in tests)
        self.use_web = use_web

    # ── entry point ──────────────────────────────────────────────────────────
    def answer(self, history):
        th = Thinking()
        asked = [t for r, t in history if r == "user"]
        q = (asked[-1] if asked else "").strip()
        previous = asked[-2] if len(asked) > 1 else ""
        lang = web.detect_language(q)
        intent = self.classify(q)
        th.step("Understanding the question", self.describe(intent, q, lang))

        if intent == "calculate":
            return th.result(calc.answer(q), "calculate")

        local, score = self.recall_match(history) if lang == "en" else (None, 0.0)
        if intent in ("chat", "code") and local is not None:
            th.step("Checking what I already know", f"I know this topic (match {score:.2f})")
            return th.result(self.recall(history, th), "recall")

        if self.use_web and intent != "chat":
            try:
                if intent == "news":
                    reply = self.news(th)
                elif intent == "define":
                    reply = self.define(q, th)
                elif intent == "rank":
                    reply = self.rank(q, th)
                else:
                    reply = None
                if reply:
                    return th.result(reply, intent)
                if local is not None and score >= 0.75:
                    th.step("Checking what I already know", f"Strong match in my trained knowledge ({score:.2f})")
                    reply = self.recall(history, th)
                    self.enrich(self.subject(q, previous), th, lang)
                    return th.result(reply, "recall")
                reply = self.lookup(q, previous, th, lang)
                if reply:
                    return th.result(reply, "look up")
            except web.WebError as e:
                th.step("Searching the web", f"Couldn't reach Wikipedia ({str(e)[:60]})")

        if local is not None:
            th.step("Checking what I already know", f"Answering from my trained knowledge (match {score:.2f})")
            return th.result(self.recall(history, th), "recall")
        th.step("Checking what I already know", "Nothing reliable found")
        from .chat import UNKNOWN_REPLY
        if self.engine is not None and self.engine.guard:
            self.engine.guard.log_unknown(q)
        return th.result(UNKNOWN_REPLY, "unknown")

    # ── understanding ────────────────────────────────────────────────────────
    @staticmethod
    def classify(q):
        if calc.answer(q):
            return "calculate"
        if NEWS.search(q):
            return "news"
        if DEFINE.match(q.strip()):
            return "define"
        if ABOUT_MAX.search(q) and not SUPERLATIVE.search(q):
            return "chat"
        if web.detect_language(q) != "en":
            return "fact"
        if not terms(q) or set(terms(q)) <= SMALL_TALK:
            return "chat"  # greetings, thanks, "ok"
        if SUPERLATIVE.search(q) and len(terms(q)) >= 1:
            return "rank"
        if CODE_WORDS.search(q):
            return "code"
        return "fact"

    @staticmethod
    def describe(intent, q, lang):
        what = {"calculate": "a calculation", "news": "a request for current news", "define": "a word definition",
                "chat": "a conversation message", "rank": "a ranking / list question", "code": "a coding question",
                "fact": "a factual question"}[intent]
        keys = ", ".join(dict.fromkeys(terms(q)))[:80]
        lang_note = f" · language: {lang}" if lang != "en" else ""
        return f"Looks like {what}" + (f" · key words: {keys}" if keys else "") + lang_note

    def recall_match(self, history):
        guard = self.engine.guard if self.engine is not None else None
        if guard is None:
            return None, 0.0
        asked = [t for r, t in history if r == "user"]
        q = asked[-1] if asked else ""
        previous = asked[-2] if len(asked) > 1 else None
        context = " ".join(asked[-3:-2] + asked[-2:-1] * 2) or None
        turns = guard.match(q, previous, context)
        score, _ = guard.score(q)
        if turns is not None and previous and turns[-1][1].strip().lower() != q.strip().lower() \
                and set(terms(turns[-1][1])) & set(terms(" ".join(asked[:-1][-2:]))):
            score = max(score, 0.8)
        return turns, score

    # ── strategies ───────────────────────────────────────────────────────────
    def recall(self, history, th):
        """MaxGPT's own answer, regenerated (cooler) if the self-check finds it garbled."""
        reply = ""
        for attempt, temp in enumerate((None, 0.35, 0.1)):
            opts = {} if temp is None else {"temperature": temp}
            reply = self.engine.reply(history, **opts)
            problem = self.quality_problem(reply)
            if not problem:
                th.step("Double-checking the answer", "Looks clear and complete" if attempt == 0 else
                        f"Fixed on try {attempt + 1}")
                return reply
            th.step("Double-checking the answer", f"{problem}, rewriting it")
        return reply

    @staticmethod
    def quality_problem(text):
        words = re.findall(r"\w+", text.lower())
        if len(text.strip()) < 2:
            return "Answer was empty"
        for n in (1, 2, 3):
            grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
            for i in range(len(grams) - 2 * n):
                if grams[i] == grams[i + n] == grams[i + 2 * n]:
                    return "Answer repeated itself"
        if text.count("```") % 2:
            return "Code block was cut off"
        odd = sum(1 for ch in text if ord(ch) == 0xFFFD)
        if odd:
            return "Answer had broken characters"
        return None

    def news(self, th):
        th.step("Checking today's news", "Wikipedia · Portal:Current events")
        items = web.news()
        if not items:
            return None
        th.step("Reading the headlines", f"{len(items)} stories")
        th.source("Wikipedia: Current events", "https://en.wikipedia.org/wiki/Portal:Current_events")
        lines = [f"{i}. {text}" for i, (_, text) in enumerate(items, 1)]
        return "Here's what's happening in the world right now:\n\n" + "\n".join(lines)

    def define(self, q, th):
        word = DEFINE.match(q.strip()).group(1).strip()
        th.step("Looking up the word", f"Wiktionary: “{word}”")
        defs = web.define(word)
        if not defs:
            return None
        th.source(f"Wiktionary: {word}", "https://en.wiktionary.org/wiki/" + word.replace(" ", "_"))
        lines = [f"{i}. ({pos.lower()}) {d}" if pos else f"{i}. {d}" for i, (pos, d) in enumerate(defs, 1)]
        return f"{word.capitalize()}:\n" + "\n".join(lines)

    def rank(self, q, th):
        want = int(re.search(r"\btop\s*(\d+)", q, re.I).group(1)) if re.search(r"\btop\s*(\d+)", q, re.I) else 10
        want = max(1, min(want, 25))
        core = re.sub(r"\b(top\s*\d+|top|list of|list|what are|which are|the|tell me|show me|give me|please|"
                      r"in the world|of all time|ever|ranking|ranked)\b", " ", q, flags=re.I)
        core = re.sub(r"[^\w\s\-']", " ", core).strip()
        core = re.sub(r"\s+", " ", core)
        queries = [f"list of {core}", core, f"{core} in the world", f"world's {core}"]
        th.step("Searching Wikipedia for rankings", " · ".join(f"“{s}”" for s in queries))
        seen, candidates = set(), []
        with ThreadPoolExecutor(4) as pool:
            for hits in pool.map(lambda s: web.search(s, limit=5), queries):
                for h in hits:
                    if h["title"] not in seen:
                        seen.add(h["title"])
                        candidates.append(h["title"])
        qt = set(terms(core))
        measure = self.measure_terms(q)
        wanted = qt | measure
        subject = qt - SUPER_WORDS - measure - {"world"}

        def title_score(t):
            tt = set(terms(t))
            return (t.lower().startswith("list of") + 2 * len(subject & tt) + (1 if measure & tt else 0)
                    - 2.5 * self.extra_words(t, wanted))
        order = {t: i for i, t in enumerate(candidates)}
        candidates.sort(key=lambda t: (-title_score(t), order[t]))
        candidates = candidates[:5]
        if not candidates:
            return None
        th.step("Reading list articles", " · ".join(candidates))
        best = None
        with ThreadPoolExecutor(4) as pool:
            pages = list(pool.map(self._safe_html, candidates))
        for title, page_html in pages:
            if not page_html:
                continue
            for table in web.tables(page_html):
                pick = self.score_table(table, q, title, wanted, subject, measure)
                if pick and (best is None or pick["score"] > best["score"]):
                    best = dict(pick, page=title)
        if best is None:
            th.step("Reading list articles", "No usable ranking table, summarising instead")
            return None
        rows = self.ranked_rows(best, want)
        if len(rows) < min(want, 5):
            th.step("Extracting the table", f"Only {len(rows)} usable rows in {best['page']}, summarising instead")
            return None
        th.step("Extracting the table",
                f"{best['page']}: {len(best['table']['rows'])} rows" +
                (f", sorted by “{best['table']['headers'][best['value_col']]}”" if best["value_col"] is not None else ""))
        th.source(f"Wikipedia: {best['page']}", web.page_url(best["page"]))
        heading = f"Here are the top {len(rows)} {core.strip()} according to Wikipedia ({best['page']}):"
        return heading + "\n\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(rows, 1)) + \
            "\n\nWant more details about any of them?"

    @staticmethod
    def measure_terms(q):
        extra = set()
        for pattern, keys, _ in MEASURES:
            if re.search(pattern, q.lower()):
                extra |= set(terms(" ".join(keys)))
        return extra

    @staticmethod
    def extra_words(title, wanted):
        """Words in a title that narrow it beyond what was asked ("... in the United Kingdom")."""
        return len(set(terms(title)) - wanted - NEUTRAL)

    @staticmethod
    def _safe_html(title):
        try:
            return web.article_html(title)
        except web.WebError:
            return title, None

    @staticmethod
    def score_table(table, q, title, qt, subject=frozenset(), measure=frozenset()):
        headers = [h.lower() for h in table["headers"]]
        rows = table["rows"]
        if len(rows) < 3 or len(headers) < 2:
            return None
        # the table must be about what was asked ("cars", not "animals")
        about = set(terms(title + " " + table["caption"] + " " + " ".join(headers)))
        if subject and not (subject & about) and not (measure & set(terms(title))):
            return None
        ql = q.lower()
        value_col, descending, measure_hit = None, True, False
        for pattern, keys, desc in MEASURES:
            if re.search(pattern, ql):
                for key in keys:
                    for c, h in enumerate(headers):
                        if key in h and not TIME_HEADER.search(h) and \
                                sum(web.number(r[c]) is not None for r in rows) >= len(rows) * 0.6:
                            value_col, descending, measure_hit = c, desc, True
                            break
                    if value_col is not None:
                        break
            if value_col is not None:
                break
        name_col = None
        for c, h in enumerate(headers):
            # a name may contain digits ("Jaguar XK120"); it just doesn't start with one
            if any(k in h for k in NAME_HEADERS) and sum(not STARTS_NUMERIC.match(r[c]) for r in rows) >= len(rows) * 0.6:
                name_col = c
                break
        if name_col is None:
            for c in range(len(headers)):
                if c != value_col and sum(not STARTS_NUMERIC.match(r[c]) and len(r[c]) > 1 for r in rows) >= len(rows) * 0.7:
                    name_col = c
                    break
        if name_col is None:
            return None
        if measure and not measure_hit:
            return None          # "fastest" needs a speed column, not a list of TV episodes
        score = math.log(len(rows) + 1) + (6 if measure_hit else 0)
        score += 2 * len(qt & set(terms(title + " " + table["caption"])))
        score -= 3 * Assistant.extra_words(title, qt)
        if title.lower().startswith("list of"):
            score += 1
        return {"score": score, "table": table, "name_col": name_col, "value_col": value_col,
                "descending": descending}

    @staticmethod
    def ranked_rows(pick, want):
        table, nc, vc = pick["table"], pick["name_col"], pick["value_col"]
        headers = table["headers"]
        aggregate = re.compile(r"^(world|total|totals|sum|average|mean|other|others|rest of the world|—|-)$", re.I)
        rows = [r for r in table["rows"] if r[nc] and not aggregate.match(r[nc].strip(" *†‡"))]
        if vc is not None:
            rows = [r for r in rows if web.number(r[vc]) is not None]
            rows.sort(key=lambda r: web.number(r[vc]), reverse=pick["descending"])
        # Give the item a fuller name when the next column continues it (e.g. Make + Model).
        extra = nc + 1 if nc + 1 < len(headers) and nc + 1 != vc and \
            any(k in headers[nc + 1].lower() for k in ("model", "name", "vehicle")) else None
        out, seen = [], set()
        for r in rows:
            name = r[nc] + (f" {r[extra]}" if extra is not None and r[extra] and r[extra] not in r[nc] else "")
            name = re.sub(r"\s+", " ", name).strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            value = f" — {r[vc][:70]}" if vc is not None else ""
            out.append(name[:90] + value)
            if len(out) >= want:
                break
        return out

    def subject(self, q, previous):
        """The search query: the question's key words, plus the earlier topic for follow-ups."""
        words = re.findall(r"[\w']+", q.lower())
        if web.detect_language(q) != "en":
            return q.strip(" ?।")                  # let that Wikipedia's own search handle the language
        query = " ".join(terms(q)) or q
        if previous and (PRONOUNS & set(words) or len(terms(q)) <= 1):
            query = " ".join(dict.fromkeys(terms(previous) + terms(q)))
        return query

    def is_follow_up(self, q, previous):
        words = set(re.findall(r"[\w']+", q.lower()))
        return bool(previous) and (bool(PRONOUNS & words) or len(terms(q)) <= 1)

    def lookup(self, q, previous, th, lang):
        follow = self.is_follow_up(q, previous) and lang == "en"
        if follow:
            # "how tall is it?" after "What is the Eiffel Tower?": find the Eiffel Tower first
            topic = " ".join(terms(previous))
            hits = web.search(topic, limit=5, lang=lang)
            th.step("Working out what “it” refers to", f"Earlier topic: “{topic}”")
            query = topic
        else:
            query = self.subject(q, previous)
            hits = web.search(query, limit=5, lang=lang)
            if not hits and query != q:
                hits = web.search(q, limit=5, lang=lang)
        th.step("Searching Wikipedia", f"{lang}.wikipedia.org · “{query}” → {len(hits)} results")
        if not hits:
            return None
        qt = set(terms(query))
        order = {h["title"]: i for i, h in enumerate(hits)}
        hits.sort(key=lambda h: (-(2 * len(qt & set(terms(h["title"]))) - 0.8 * self.extra_words(h["title"], qt)
                                   + 0.2 * len(qt & set(terms(h["snippet"])))), order[h["title"]]))
        titles = [h["title"] for h in hits[:1 if follow else 3]]
        pages = web.articles(titles, lang=lang)
        # keep the main article, plus others only if they're about the same thing
        pages = pages[:1] + [p for p in pages[1:] if self.extra_words(p["title"], qt) <= 1]
        th.step("Reading articles", " · ".join(p["title"] for p in pages) or "none readable")
        if not pages:
            return None
        top = pages[0]
        if len(qt & set(terms(top["title"] + " " + top["extract"][:400]))) == 0:
            th.step("Checking relevance", "The articles don't match the question")
            return None
        self.set_card(top, th)
        for p in pages:
            th.source(f"Wikipedia: {p['title']}", p["url"])
        context = query if not follow else query + " " + " ".join(terms(q))
        sentences = self.best_sentences(q, pages, context=context)
        # If the intro doesn't answer it, read the whole main article.
        answered = any(set(terms(s)) & set(terms(q)) for s, _ in sentences)
        if (follow or self.needs_detail(q)) and not (answered and sentences and sentences[0][1] >= 2):
            full = web.articles([top["title"]], lang=lang, intro=False)
            if full:
                th.step("Reading the full article", top["title"])
                sentences = self.best_sentences(q, [dict(top, extract=full[0]["extract"])] + pages[1:], context=context)
        th.step("Picking the sentences that answer it", f"{len(sentences)} sentences from {len(pages)} articles")
        return " ".join(s for s, _ in sentences)

    @staticmethod
    def needs_detail(q):
        return bool(re.search(r"\b(when|how many|how much|how old|how long|how tall|how far|where|which year|"
                              r"who (founded|invented|discovered|created|wrote|directed|owns))\b", q, re.I))

    @staticmethod
    def split_sentences(text):
        text = re.sub(r"\s*\n+\s*", " ", text)
        text = re.sub(r"\s*\([^()]{0,80}\)", "", text)  # pronunciations and asides
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?<=[।。！？])\s*", text)
        return [p.strip() for p in parts if 25 <= len(p.strip()) <= 400]

    def best_sentences(self, q, pages, limit=4, max_chars=700, context=""):
        # the question's own words count most; the topic carried over from earlier helps a little
        weights = {t: 0.5 for t in terms(context)}
        weights.update({t: 2.0 for t in terms(q)})
        wants_number = bool(re.search(r"\b(when|how many|how much|how old|how long|how tall|how far|year|date)\b", q, re.I))
        scored = []
        for pi, p in enumerate(pages):
            for si, s in enumerate(self.split_sentences(p["extract"])):
                st = set(terms(s))
                score = sum(w for t, w in weights.items() if t in st)
                if wants_number and re.search(r"\d", s):
                    score += 1
                score -= pi * 0.5 + si * 0.02
                scored.append((pi, si, s, score))
        if not scored:
            return []
        first = next(x for x in scored if x[0] == 0)            # definition sentence of the best article
        rest = sorted((x for x in scored if x is not first and x[3] > 0), key=lambda x: -x[3])
        # A precise question ("when did he found SpaceX?") leads with the sentence that answers it.
        precise = self.needs_detail(q) or web.detect_language(q) != "en"
        if precise and rest and rest[0][3] > max(first[3], 0) + 0.5:
            first, rest = rest[0], rest[1:]
            limit = min(limit, 2)
        chosen, size = [first], len(first[2])
        for x in rest:
            if len(chosen) >= limit or size + len(x[2]) > max_chars:
                break
            if all(x[2][:60] != c[2][:60] for c in chosen):
                chosen.append(x)
                size += len(x[2])
        lead = chosen[0]
        chosen = [lead] + sorted(chosen[1:], key=lambda x: (x[0], x[1]))
        return [(x[2], x[3]) for x in chosen]

    def enrich(self, query, th, lang="en"):
        """Add a knowledge card and source for a topic answered from memory (best effort)."""
        try:
            hits = web.search(query, limit=3, lang=lang)
            if not hits:
                return
            pages = web.articles([hits[0]["title"]], lang=lang)
        except web.WebError:
            return
        qt = set(terms(query))
        title = set(terms(pages[0]["title"])) if pages else set()
        if pages and len(qt & title) >= min(2, len(qt)) and self.extra_words(pages[0]["title"], qt) <= 1:
            self.set_card(pages[0], th)
            th.source(f"Wikipedia: {pages[0]['title']}", pages[0]["url"])
            th.step("Adding a knowledge card", pages[0]["title"])

    @staticmethod
    def set_card(page, th):
        first = Assistant.split_sentences(page["extract"])
        th.card = {"title": page["title"], "description": page.get("description", ""),
                   "summary": " ".join(first[:2])[:320], "image": page.get("image"), "url": page["url"]}
