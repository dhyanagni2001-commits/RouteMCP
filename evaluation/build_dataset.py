#!/usr/bin/env python3
"""Generate the RouteMCP routing evaluation set with deterministic, checkable answers.

Every expected answer is computed by this script (math is evaluated exactly, code
outputs are produced by running the generated snippet), so the dataset needs no
human labels and can be regenerated bit-for-bit:

    python evaluation/build_dataset.py --output evaluation/tasks.jsonl --seed 7

Task types and difficulty are deliberately mixed: cheap models should handle the
easy rows, and the hard rows should separate stronger models.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from fractions import Fraction
from typing import Any, Callable, Dict, List

FIRST = ["Ada", "Grace", "Alan", "Linus", "Barbara", "Edsger", "Margaret", "Ken", "Radia", "Donald",
         "Frances", "Dennis", "Katherine", "Tim", "Hedy", "John", "Shafi", "Leslie", "Fei", "Yukihiro"]
LAST = ["Lovelace", "Hopper", "Turing", "Torvalds", "Liskov", "Dijkstra", "Hamilton", "Thompson",
        "Perlman", "Knuth", "Allen", "Ritchie", "Johnson", "Berners", "Lamarr", "McCarthy",
        "Goldwasser", "Lamport", "Li", "Matsumoto"]
CITIES = ["Lisbon", "Nairobi", "Osaka", "Denver", "Krakow", "Montreal", "Santiago", "Hanoi",
          "Accra", "Auckland", "Bergen", "Cusco", "Tallinn", "Austin", "Porto", "Seoul"]
WORDS = ["river", "lantern", "copper", "meadow", "violet", "anchor", "pepper", "glacier", "harbor",
         "falcon", "orchid", "saddle", "timber", "velvet", "walnut", "zephyr", "basil", "cobalt",
         "dune", "ember", "fjord", "garnet", "hazel", "iris", "juniper", "kelp", "lichen", "maple"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

POSITIVE = [
    "The setup took two minutes and everything just worked. I'm thrilled.",
    "Support replied within an hour and fixed my issue. Fantastic service.",
    "This is the most reliable laptop I have ever owned; I love it.",
    "The hotel staff were warm and the room was spotless. Wonderful stay.",
    "Clear docs, fast builds, and a friendly community. Highly recommend.",
    "My order arrived early and the quality exceeded my expectations.",
    "The new update made the app faster and much easier to use. Great job.",
    "Delicious food, generous portions, and a lovely atmosphere.",
    "I was nervous about the course, but it was engaging and worth every penny.",
    "The repair was quick, affordable, and the bike rides like new.",
]
NEGATIVE = [
    "The package arrived broken and nobody answered my emails. Awful.",
    "After the update the app crashes every time I open it. Terrible.",
    "The room smelled of smoke and the staff were rude. Never again.",
    "It stopped charging after a week and the refund was refused.",
    "Slow, buggy, and the documentation is wrong in half the places.",
    "We waited ninety minutes for cold food. A complete disappointment.",
    "The headphones hurt my ears and the sound is muddy. Waste of money.",
    "Customer service hung up on me twice. I'm furious.",
    "The course was disorganized and the instructor never showed up.",
    "My flight was cancelled without notice and the airline offered nothing.",
]
TOPICS: Dict[str, List[str]] = {
    "sports": [
        "The striker scored twice in extra time to win the cup final.",
        "She broke the national record in the 400 metre hurdles on Saturday.",
        "The team traded its starting pitcher before the deadline.",
        "After five sets, the defending champion was knocked out in the quarterfinal.",
        "The marathon route will pass the stadium twice this year.",
    ],
    "finance": [
        "The central bank raised interest rates by a quarter point.",
        "Shares fell eight percent after the company cut its earnings forecast.",
        "The bond yield curve inverted for the first time since spring.",
        "Quarterly revenue beat analyst estimates, lifting the stock.",
        "The startup closed a forty million dollar series B round.",
    ],
    "technology": [
        "The new chip doubles inference throughput while using less power.",
        "A patch for the critical browser vulnerability ships tomorrow.",
        "The open-source database added native vector search in its latest release.",
        "Engineers migrated the service to a new container orchestration platform.",
        "The phone's operating system update adds on-device speech recognition.",
    ],
    "weather": [
        "Forecasters expect heavy snow across the northern valleys overnight.",
        "A heat advisory is in effect with temperatures above forty degrees.",
        "The hurricane weakened to a tropical storm before landfall.",
        "Dense fog will reduce visibility on the coast until late morning.",
        "Thunderstorms with large hail are possible this afternoon.",
    ],
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}
CURRENCY_TOOL = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Convert an amount between currencies using ISO 4217 codes.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
                "to_currency": {"type": "string", "description": "ISO 4217 code, e.g. EUR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
}
REMINDER_TOOL = {
    "type": "function",
    "function": {
        "name": "create_reminder",
        "description": "Create a reminder that fires after a number of minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "minutes": {"type": "integer"},
            },
            "required": ["text", "minutes"],
        },
    },
}
CURRENCIES = {"US dollars": "USD", "euros": "EUR", "Japanese yen": "JPY", "British pounds": "GBP",
              "Swiss francs": "CHF", "Canadian dollars": "CAD", "Indian rupees": "INR"}


def user(content: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": content}]


def number_text(value: Fraction) -> float:
    return float(value) if value.denominator != 1 else int(value)


def classification(rng: random.Random) -> List[Dict[str, Any]]:
    rows = []
    for index, (text, label) in enumerate(
        [(t, "positive") for t in POSITIVE] + [(t, "negative") for t in NEGATIVE]
    ):
        rows.append({
            "id": f"classification-sentiment-{index:03d}",
            "task_type": "classification",
            "difficulty": "easy",
            "messages": user(
                "Classify the sentiment of this review as positive or negative. "
                f"Reply with one word.\n\nReview: {text}"
            ),
            "expected": {"type": "choice", "value": label, "choices": ["positive", "negative"]},
        })
    topics = sorted(TOPICS)
    index = 0
    for topic in topics:
        for text in TOPICS[topic]:
            rows.append({
                "id": f"classification-topic-{index:03d}",
                "task_type": "classification",
                "difficulty": "easy",
                "messages": user(
                    "Classify this headline into exactly one category: "
                    f"{', '.join(topics)}. Reply with the category name only.\n\nHeadline: {text}"
                ),
                "expected": {"type": "choice", "value": topic, "choices": topics},
            })
            index += 1
    return rows


def extraction(rng: random.Random) -> List[Dict[str, Any]]:
    rows = []
    for index in range(40):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        age, city = rng.randint(21, 68), rng.choice(CITIES)
        email = f"{first.lower()}.{last.lower()}@example.com"
        if index < 20:
            text = (f"Hi all, {first} {last} here. I'm {age}, I recently moved to {city}, "
                    f"and you can reach me at {email}.")
            prompt = ("Extract the person's details from the message as a JSON object with keys "
                      "name, age (integer), city, and email.")
            difficulty = "easy"
        else:
            other_first, other_last = rng.choice(FIRST), rng.choice(LAST)
            other_age = age + rng.choice([-9, -4, 3, 7, 12])
            other_city = rng.choice([c for c in CITIES if c != city])
            text = (f"Onboarding note: {other_first} {other_last} ({other_age}, based in "
                    f"{other_city}) will pair with the new manager. The manager is {age} years "
                    f"old, lives in {city}, goes by {first} {last}, and uses {email}. "
                    f"{other_first}'s address is {other_first.lower()}@example.org.")
            prompt = ("Extract the MANAGER's details (not the other person's) as a JSON object "
                      "with keys name, age (integer), city, and email.")
            difficulty = "hard"
        rows.append({
            "id": f"extraction-{index:03d}",
            "task_type": "extraction",
            "difficulty": difficulty,
            "messages": user(f"{prompt} Reply with JSON only.\n\n{text}"),
            "response_format": {"type": "json_object"},
            "expected": {
                "type": "json_match",
                "value": {"name": f"{first} {last}", "age": age, "city": city, "email": email},
            },
        })
    return rows


def transformation(rng: random.Random) -> List[Dict[str, Any]]:
    rows = []
    for index in range(40):
        kind = index % 4
        words = rng.sample(WORDS, rng.randint(5, 9))
        if kind == 0:
            prompt = "Rewrite these words in UPPERCASE, separated by single spaces"
            answer, difficulty = " ".join(w.upper() for w in words), "easy"
        elif kind == 1:
            prompt = "Rewrite this list of words in reverse order, separated by single spaces"
            answer, difficulty = " ".join(reversed(words)), "medium"
        elif kind == 2:
            prompt = "Rewrite these words sorted alphabetically, separated by commas and a space"
            answer, difficulty = ", ".join(sorted(words)), "medium"
        else:
            prompt = ("Rewrite each word with its letters reversed, keeping the word order, "
                      "separated by single spaces")
            answer, difficulty = " ".join(w[::-1] for w in words), "hard"
        rows.append({
            "id": f"transformation-{index:03d}",
            "task_type": "transformation",
            "difficulty": difficulty,
            "messages": user(f"{prompt}. Reply with the result only.\n\n{' '.join(words)}"),
            "expected": {"type": "normalized", "value": answer},
        })
    return rows


def math(rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    easy: List[Callable[[], tuple]] = [
        lambda: (lambda a, b: (f"A shelf holds {a} books and another holds {b}. "
                               "How many books are there in total?", Fraction(a + b)))(
            rng.randint(12, 90), rng.randint(12, 90)),
        lambda: (lambda a, b: (f"A crate has {a} rows of {b} apples. How many apples is that?",
                               Fraction(a * b)))(rng.randint(4, 15), rng.randint(4, 15)),
        lambda: (lambda a, b: (f"A tank holds {a} litres and {b} litres are drained. "
                               "How many litres remain?", Fraction(a - b)))(
            rng.randint(200, 500), rng.randint(20, 190)),
    ]
    hard: List[Callable[[], tuple]] = [
        lambda: (lambda a, p, b, c: (
            f"A warehouse has {a} boxes. It ships {p}% of them on Monday, ships {b} more on "
            f"Tuesday, and receives {c} new boxes on Wednesday. How many boxes does it have now?",
            Fraction(a) - Fraction(a * p, 100) - b + c))(
            rng.randrange(200, 1000, 20), rng.choice([5, 10, 15, 25, 35]), rng.randint(10, 60),
            rng.randint(5, 80)),
        lambda: (lambda x, y, a, c, d: (
            f"Adult tickets cost ${x} and child tickets cost ${y}. A group buys {a} adult and "
            f"{c} child tickets and gets a {d}% discount on the total. How much do they pay in "
            "dollars?", (Fraction(x * a + y * c) * (100 - d)) / 100))(
            rng.randint(12, 40), rng.randint(5, 11), rng.randint(2, 7), rng.randint(1, 6),
            rng.choice([10, 15, 20])),
        lambda: (lambda d1, t1, d2, t2: (
            f"A cyclist rides {d1} km in {t1} hour{'s' * (t1 > 1)}, rests, then rides {d2} km "
            f"in {t2} hour{'s' * (t2 > 1)}. "
            "What is the average speed in km/h over the riding time only? Round to two decimals "
            "if needed.", Fraction(d1 + d2, t1 + t2)))(
            rng.randint(20, 60), rng.randint(1, 3), rng.randint(15, 50), rng.randint(1, 3)),
        lambda: (lambda n, k, m: (
            f"A loop counts every integer from 1 to {n} that is divisible by {k} but not by {m}. "
            "How many integers does it count?",
            Fraction(sum(1 for i in range(1, n + 1) if i % k == 0 and i % m != 0))))(
            rng.randint(150, 400), rng.choice([3, 4, 6]), rng.choice([5, 7, 9])),
    ]
    for index in range(50):
        if index < 20:
            question, answer = easy[index % len(easy)]()
            difficulty = "easy"
        else:
            question, answer = hard[index % len(hard)]()
            difficulty = "hard"
        tolerance = 0.01 if answer.denominator != 1 else 1e-6
        rows.append({
            "id": f"math-{index:03d}",
            "task_type": "math",
            "difficulty": difficulty,
            "messages": user(f"{question} Answer with only the final number."),
            "expected": {"type": "number", "value": number_text(answer), "tolerance": tolerance},
        })
    return rows


def reasoning(rng: random.Random) -> List[Dict[str, Any]]:
    rows = []
    for index in range(30):
        if index % 2 == 0:
            names = rng.sample(FIRST, rng.choice([4, 5]) if index < 14 else 6)
            facts = [f"{names[i]} is taller than {names[i + 1]}." for i in range(len(names) - 1)]
            rng.shuffle(facts)
            ask_tallest = rng.random() < 0.5
            answer = names[0] if ask_tallest else names[-1]
            question = "Who is the tallest?" if ask_tallest else "Who is the shortest?"
            rows.append({
                "id": f"reasoning-order-{index:03d}",
                "task_type": "reasoning",
                "difficulty": "easy" if len(names) <= 4 else "hard",
                "messages": user(f"{' '.join(facts)} {question} Reply with the name only."),
                "expected": {"type": "choice", "value": answer, "choices": names},
            })
        else:
            start = rng.randrange(7)
            offset = rng.randint(3, 12) if index < 14 else rng.randint(40, 400)
            answer = DAYS[(start + offset) % 7]
            rows.append({
                "id": f"reasoning-days-{index:03d}",
                "task_type": "reasoning",
                "difficulty": "easy" if offset < 13 else "hard",
                "messages": user(
                    f"Today is {DAYS[start]}. What day of the week will it be {offset} days "
                    "from today? Reply with the day name only."
                ),
                "expected": {"type": "choice", "value": answer, "choices": DAYS},
            })
    return rows


def _run(source: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(source, {})  # noqa: S102 - executes snippets generated by this script only
    return buffer.getvalue().strip()


def coding(rng: random.Random) -> List[Dict[str, Any]]:
    templates: List[Callable[[], tuple]] = [
        lambda: ("easy", "values = {v}\nprint(sum(x * x for x in values))".format(
            v=rng.sample(range(1, 12), 4))),
        lambda: ("easy", 'word = "{w}"\nprint(word[::-1].upper())'.format(w=rng.choice(WORDS))),
        lambda: ("medium", "total = 0\nfor i in range({n}):\n    if i % {k} == 0:\n"
                 "        total += i\nprint(total)".format(n=rng.randint(15, 40), k=rng.choice([3, 4, 5]))),
        lambda: ("medium", 'counts = {{}}\nfor ch in "{w}":\n    counts[ch] = counts.get(ch, 0) + 1\n'
                 "print(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0])".format(
                     w=rng.choice(["mississippi", "bookkeeper", "abracadabra", "committee",
                                   "balloon", "assessment"]))),
        lambda: ("hard", "def f(n):\n    if n <= 1:\n        return n\n    return f(n - 1) + f(n - 2)\n"
                 "print([f(i) for i in range({a}, {b})])".format(
                     a=(a := rng.randint(3, 7)), b=a + rng.randint(3, 5))),
        lambda: ("hard", "data = {d}\nresult = []\nfor x in data:\n    if x % 2:\n"
                 "        result.insert(0, x * {m})\n    else:\n        result.append(x // 2)\n"
                 "print(result)".format(d=rng.sample(range(1, 20), 6), m=rng.choice([2, 3]))),
    ]
    rows = []
    for index in range(30):
        difficulty, source = templates[index % len(templates)]()
        rows.append({
            "id": f"coding-{index:03d}",
            "task_type": "coding",
            "difficulty": difficulty,
            "messages": user(
                "What does this Python program print? Reply with the exact output only, "
                f"no explanation.\n\n```python\n{source}\n```"
            ),
            "expected": {"type": "normalized", "value": _run(source)},
        })
    return rows


def tool_calling(rng: random.Random) -> List[Dict[str, Any]]:
    tools = [WEATHER_TOOL, CURRENCY_TOOL, REMINDER_TOOL]
    rows = []
    for index in range(30):
        kind = index % 3
        if kind == 0:
            city, unit = rng.choice(CITIES), rng.choice(["celsius", "fahrenheit"])
            prompt = f"What's the weather like in {city} right now? Give it to me in {unit}."
            spec = {"name": "get_weather", "arguments": {"city": city, "unit": unit}}
            difficulty = "easy"
        elif kind == 1:
            source, target = rng.sample(sorted(CURRENCIES), 2)
            amount = rng.randrange(20, 5000, 5)
            prompt = f"How much is {amount} {source} in {target}?"
            spec = {
                "name": "convert_currency",
                "arguments": {"amount": amount, "from_currency": CURRENCIES[source],
                              "to_currency": CURRENCIES[target]},
            }
            difficulty = "medium"
        else:
            hours, minutes = rng.randint(1, 3), rng.choice([0, 15, 30, 45])
            task = rng.choice(["stretch", "call the dentist", "check the oven", "submit the report"])
            when = f"{hours} hour{'s' if hours > 1 else ''}" + (f" and {minutes} minutes" if minutes else "")
            prompt = f"Remind me to {task} in {when}."
            spec = {"name": "create_reminder", "arguments": {"minutes": hours * 60 + minutes}}
            difficulty = "hard"
        rows.append({
            "id": f"tool-{index:03d}",
            "task_type": "tool_calling",
            "difficulty": difficulty,
            "messages": user(prompt),
            "tools": tools,
            "expected": {"type": "tool_call", "value": spec},
        })
    return rows


def build(seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for generator in (classification, extraction, transformation, math, reasoning, coding, tool_calling):
        rows.extend(generator(rng))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evaluation/tasks.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows = build(args.seed)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()
