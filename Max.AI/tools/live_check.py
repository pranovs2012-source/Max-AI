"""Ask Max real questions against live Wikipedia and print the answers with their thinking steps.

    python tools/live_check.py            (from the Max.AI folder)

Exits non-zero if a core ability is broken, so CI catches it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxgpt.chat import MaxGPTEngine  # noqa: E402
from maxgpt.think import Assistant  # noqa: E402

CONVERSATIONS = [
    ["top 10 fastest cars"],
    ["tallest buildings in the world"],
    ["most populous countries"],
    ["top 5 richest people"],
    ["Who is Elon Musk?", "when did he found SpaceX?"],
    ["What is the Eiffel Tower?", "how tall is it?"],
    ["Tell me about black holes"],
    ["What is the capital of Japan?", "And its currency?"],
    ["latest news"],
    ["define serendipity"],
    ["भारत की राजधानी क्या है"],
    ["Who are you?"],
    ["How do I reverse a string in python"],
    ["What is 15% of 240?"],
]

assistant = Assistant(MaxGPTEngine())
failures, summary = [], []
for convo in CONVERSATIONS:
    history, result = [], None
    for q in convo:
        history.append(("user", q))
        result = assistant.answer(history)
        history.append(("assistant", result["reply"]))
    print("=" * 100)
    print("Q:", " → ".join(convo), f"   [{result['route']}, {result['seconds']}s]")
    print(result["reply"])
    for s in result["thinking"]:
        print(f"   · {s['title']}: {s['detail']}  ({s['ms']} ms)")
    if result["sources"]:
        print("   sources:", ", ".join(s["url"] for s in result["sources"]))
    if result["card"]:
        print("   card:", result["card"]["title"], "|", result["card"].get("image"))
    lines = [l for l in result["reply"].splitlines() if l.strip()]
    source = result["sources"][0]["title"] if result["sources"] else "-"
    read = next((s["detail"] for s in result["thinking"] if s["title"].startswith("Reading")), "")
    summary.append(f"[{result['route']:>9}] {' → '.join(convo)[:60]:60} | {source[:45]:45} | {' / '.join(lines[:3])[:150]}")
    if read:
        summary.append(" " * 12 + "read: " + read[:200])
    if convo[0] == "top 10 fastest cars" and sum(l[:3].rstrip(".").isdigit() for l in result["reply"].splitlines()) < 5:
        failures.append("ranking list")
    if convo[0] == "Who is Elon Musk?" and "2002" not in result["reply"]:
        failures.append("follow-up lookup")
    if convo[0] == "latest news" and result["route"] != "news":
        failures.append("news")
    if convo[0] == "most populous countries" and not ("India" in result["reply"] and "China" in result["reply"]):
        failures.append("world population ranking")
    if convo[0] == "tallest buildings in the world" and "Burj Khalifa" not in result["reply"]:
        failures.append("world buildings ranking")
    if convo[0].startswith("भारत") and ("दिल्ली" not in result["reply"] and "Delhi" not in result["reply"]):
        failures.append("Hindi question")
    if convo[-1] == "And its currency?" and "yen" not in result["reply"].lower():
        failures.append("follow-up from memory")

print("=" * 100)
print("SUMMARY")
print("\n".join(summary))
if failures:
    print("FAILED:", ", ".join(failures))
    sys.exit("Broken: " + ", ".join(failures))
print("All core checks passed.")
