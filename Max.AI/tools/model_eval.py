"""Grade a MaxGPT checkpoint on what a basic assistant must do, using the model alone (no guard,
no web): know who it is, answer everyday questions in sentences, answer from search results,
and admit when the results don't contain the answer.

    python tools/model_eval.py [checkpoint_dir]      (from the Max.AI folder)

Prints every answer and a score from 0 to 1; writes it to <checkpoint_dir>/eval.json.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxgpt.chat import DEFAULT_DIR, MaxGPTEngine  # noqa: E402
from maxgpt.data import grounded_question  # noqa: E402
from maxgpt.think import Assistant  # noqa: E402

CHAT = [  # (question, any of these words must appear)
    ("Who are you?", ["max"]),
    ("Who created you?", ["pranov"]),
    ("What is your name?", ["max"]),
    ("Hello!", ["hi", "hello", "hey", "help"]),
    ("What is photosynthesis?", ["plant", "light", "sun"]),
    ("What is the capital of France?", ["paris"]),
    ("How do I reverse a string in Python?", ["[::-1]", "reverse"]),
    ("Give me three tips for staying focused.", ["focus", "break", "phone", "distract"]),
]

GROUNDED = [  # (question, passages, expected word)
    ("How tall is the Eiffel Tower?",
     ["The Eiffel Tower is a wrought-iron lattice tower in Paris, France. The tower is 330 metres tall, about "
      "the same height as an 81-storey building, and was the tallest man-made structure in the world until 1930.",
      "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor."], "330"),
    ("When was SpaceX founded?",
     ["Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American spacecraft manufacturer. "
      "It was founded in 2002 by Elon Musk with the goal of reducing space transportation costs."], "2002"),
    ("What is the capital of Australia?",
     ["Australia is a country made up of the mainland of the Australian continent, the island of Tasmania and "
      "numerous smaller islands. Its capital is Canberra, and its largest city is Sydney."], "canberra"),
    ("Who wrote Romeo and Juliet?",
     ["Paris is the capital and largest city of France.",
      "Romeo and Juliet is a tragedy written by William Shakespeare early in his career about the romance "
      "between two Italian youths from feuding families."], "shakespeare"),
    ("How many players are on a football team on the field?",
     ["Association football is a team sport played between two teams of 11 players who almost exclusively use "
      "their feet to propel a ball around a rectangular field called a pitch."], "11"),
    ("What is the boiling point of water at sea level?",
     ["Water is an inorganic compound with the chemical formula H2O. At sea level, pure water boils at "
      "100 degrees Celsius (212 degrees Fahrenheit)."], "100"),
]

NO_ANSWER = [
    ("Who won the 1998 chess olympiad?",
     ["The giraffe is a large African hoofed mammal. It is the tallest living terrestrial animal.",
      "Bread is a staple food prepared from a dough of flour and water, usually by baking."]),
]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    eng = MaxGPTEngine(ckpt, temperature=0.0, guard=False)
    print(f"model: {eng.model.num_params():,} parameters · grounded={eng.grounded} · {eng.meta.get('minutes', 0):.0f} min trained")
    results = []

    def ask(text):
        return eng.reply([("user", text)], temperature=0.0, max_new_tokens=120)

    for q, words in CHAT:
        a = ask(q)
        ok = any(w in a.lower() for w in words) and not Assistant.quality_problem(a)
        results.append(ok)
        print(f"[{'PASS' if ok else 'fail'}] chat     {q!r} -> {a!r}")
    for q, passages, word in GROUNDED:
        a = ask(grounded_question(q, passages))
        ok = word in a.lower() and not Assistant.quality_problem(a)
        results.append(ok)
        print(f"[{'PASS' if ok else 'fail'}] grounded {q!r} -> {a!r}")
    for q, passages in NO_ANSWER:
        a = ask(grounded_question(q, passages))
        ok = bool(re.search(r"couldn't find|don't say|not sure|rather not", a.lower()))
        results.append(ok)
        print(f"[{'PASS' if ok else 'fail'}] no-answer {q!r} -> {a!r}")
    score = sum(results) / len(results)
    chat = sum(results[:len(CHAT)]) / len(CHAT)
    reading = sum(results[len(CHAT):]) / (len(results) - len(CHAT))
    print(f"SCORE {score:.2f} ({sum(results)}/{len(results)}) · chat {chat:.2f} · reading {reading:.2f}")
    json.dump({"score": score, "chat": chat, "reading": reading, "passed": sum(results), "total": len(results)},
              open(os.path.join(ckpt, "eval.json"), "w"))


if __name__ == "__main__":
    main()
