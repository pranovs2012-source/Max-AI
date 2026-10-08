"""Tests for Max's reasoning loop, the Wikipedia table reader and the voice text, with a fake web."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxgpt import web  # noqa: E402
from maxgpt.think import Assistant, terms  # noqa: E402

SPEED_TABLE = """<table class="wikitable sortable"><caption>Top speeds</caption>
<tr><th rowspan="2">Make</th><th rowspan="2">Model</th><th colspan="2">Top speed</th></tr>
<tr><th>km/h</th><th>mph</th></tr>
<tr><td rowspan="2">Bugatti</td><td>Veyron<sup class="reference">[1]</sup></td><td>407</td><td>253</td></tr>
<tr><td>Chiron Super Sport 300+</td><td><span style="display:none">0490</span>490.48</td><td>304.77</td></tr>
<tr><td>SSC</td><td>Tuatara</td><td>455.3</td><td>282.9</td></tr>
<tr><td>Koenigsegg</td><td>Agera RS</td><td>447.19</td><td>277.87</td></tr>
</table>"""

ARTICLES = {
    "Elon Musk": "Elon Reeve Musk (born June 28, 1971) is a businessman known for his roles in Tesla and SpaceX. "
                 "He was born in Pretoria, South Africa. Musk founded SpaceX in 2002 and joined Tesla in 2004.",
}


def fake_search(query, limit=6, lang="en"):
    q = query.lower()
    if "fast" in q:
        return [{"title": "Production car speed record", "snippet": ""}]
    if "musk" in q or "spacex" in q:
        return [{"title": "Elon Musk", "snippet": "businessman"}]
    return []


def fake_articles(titles, lang="en", intro=True):
    return [{"title": t, "extract": ARTICLES[t], "description": "Businessman", "url": web.page_url(t), "image": None}
            for t in titles if t in ARTICLES]


def fake_html(title, lang="en"):
    return title, SPEED_TABLE


FAKE_WEB = dict(search=fake_search, articles=fake_articles, article_html=fake_html,
                news=lambda days=2, limit=8: [("2026-10-08", "Leaders met in Geneva to agree on new climate targets.")],
                define=lambda word: [("Noun", "A happy accident.")],
                web_search=lambda query, limit=5: [])


class TableTest(unittest.TestCase):
    def test_spans_hidden_text_and_references(self):
        [table] = web.tables(SPEED_TABLE)
        self.assertEqual(table["headers"], ["Make", "Model", "Top speed / km/h", "Top speed / mph"])
        self.assertEqual(table["rows"][1], ["Bugatti", "Chiron Super Sport 300+", "490.48", "304.77"])
        self.assertEqual(table["rows"][0][1], "Veyron")

    def test_numbers(self):
        self.assertEqual(web.number("490.48 km/h (304.77 mph)"), 490.48)
        self.assertEqual(web.number("US$1.5 billion"), 1.5e9)
        self.assertIsNone(web.number("—"))

    def test_language(self):
        self.assertEqual(web.detect_language("भारत की राजधानी क्या है"), "hi")
        self.assertEqual(web.detect_language("What is the capital of India?"), "en")


@mock.patch.multiple(web, **FAKE_WEB)
class AssistantTest(unittest.TestCase):
    def ask(self, *questions):
        history, result = [], None
        for q in questions:
            history.append(("user", q))
            result = Assistant(engine=None).answer(history)
            history.append(("assistant", result["reply"]))
        return result

    def test_routes(self):
        cases = {"top 10 fastest cars": "rank", "latest news": "news", "define serendipity": "define",
                 "what is 6*7": "calculate", "Who is Elon Musk?": "fact", "hi": "chat",
                 "How do I reverse a string in python": "code", "most populous cities": "rank"}
        for q, intent in cases.items():
            self.assertEqual(Assistant.classify(q), intent, q)

    def test_ranking_reads_and_sorts_the_table(self):
        r = self.ask("top 3 fastest cars")
        self.assertEqual(r["route"], "rank")
        lines = [l for l in r["reply"].splitlines() if l[:2] in ("1.", "2.", "3.")]
        self.assertTrue(lines[0].startswith("1. **Bugatti Chiron Super Sport 300+** — 490.48"), lines)
        self.assertTrue(lines[1].startswith("2. **SSC Tuatara**"), lines)
        self.assertIn("ranked by top speed", r["reply"])
        self.assertNotIn("Wikipedia", r["reply"])
        self.assertEqual(len(lines), 3)
        self.assertIn("Production car speed record", r["sources"][0]["title"])

    def test_lookup_with_follow_up(self):
        r = self.ask("Who is Elon Musk?")
        self.assertEqual(r["route"], "look up")
        self.assertTrue(r["reply"].startswith("**Elon Reeve Musk** is a businessman"), r["reply"])
        self.assertEqual(r["card"]["title"], "Elon Musk")
        r = self.ask("Who is Elon Musk?", "when did he found SpaceX?")
        self.assertTrue(r["reply"].startswith("Musk founded SpaceX in 2002"), r["reply"])

    def test_news_define_and_unknown(self):
        self.assertIn("Geneva", self.ask("what's the latest news?")["reply"])
        self.assertEqual(self.ask("define serendipity")["reply"], "**Serendipity** (noun) means a happy accident.")
        r = self.ask("who won the 2030 world cup")
        self.assertEqual(r["route"], "unknown")
        self.assertTrue(r["thinking"])

    def test_web_down_is_handled(self):
        def down(*a, **k):
            raise web.WebError("offline")
        with mock.patch.object(web, "search", down):
            self.assertEqual(self.ask("Who is Elon Musk?")["route"], "unknown")

    def test_written_answers_must_match_the_sources(self):
        ctx = "The tower is 330 metres tall and was completed in 1889."
        self.assertIsNone(Assistant.unsupported("The tower is 330 metres tall.", ctx))
        self.assertIsNotNone(Assistant.unsupported("The tower is 350 metres tall.", ctx))        # wrong number
        self.assertIsNotNone(Assistant.unsupported("The tower is painted bright purple daily.", ctx))  # made up
        self.assertIsNotNone(Assistant.unsupported("I couldn't find the answer to that.", ctx))

    def test_copied_drafts_are_detected(self):
        passages = ["The tower is 330 metres tall, about the same height as an 81-storey building."]
        self.assertTrue(Assistant.copied("The tower is 330 metres tall, about the same height as an 81-storey building.", passages))
        self.assertFalse(Assistant.copied("The Eiffel Tower is 330 metres tall.", passages))

    def test_self_check(self):
        self.assertIsNotNone(Assistant.quality_problem("Pranov. Pranov. Pranov. Pranov."))
        self.assertIsNotNone(Assistant.quality_problem("```python\nprint(1)"))
        self.assertIsNone(Assistant.quality_problem("The capital of Japan is Tokyo."))

    def test_terms(self):
        self.assertEqual(terms("most populous cities"), ["populous", "city"])


class VoiceTextTest(unittest.TestCase):
    def test_speakable(self):
        import voice
        text = voice.speakable("Use **slicing**:\n```python\nprint(x[::-1])\n```\nSee https://example.com now.")
        self.assertNotIn("print", text)
        self.assertNotIn("http", text)
        self.assertIn("code on your screen", text)
        self.assertLessEqual(len(voice.speakable("Word. " * 400)), voice.MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
