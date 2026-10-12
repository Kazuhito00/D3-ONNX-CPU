# -*- coding: utf-8 -*-
"""Request set shared by the golden generator and the verification gates (deterministic, no model needed)."""
from __future__ import annotations

import random

ROUTE = {"returns": "Refunds, replacements and damaged deliveries",
         "billing": "Payments, invoices and charges",
         "technical": "Product setup and faults"}

STATES_EN = [
    "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    "I was charged twice for my Pro subscription this month and nobody answers my emails.",
    "The app crashes every time I open the settings page on Android 14.",
    "Please summarise the quarterly numbers: revenue rose 12 percent, costs fell 3 percent, headcount flat.",
    "Dear team, the server room temperature alarm went off at 02:14 and the backup job failed twice.",
    "Thanks for the quick fix, everything works again and the customer is very happy.",
    "def add(a, b):\n    return a - b  # expected the sum",
    "The patient reports mild headache for two days, no fever, no vomiting, normal appetite.",
    "Tokyo is the capital of Japan. Mount Fuji is 3,776 metres tall and lies west of the city.",
    "",
]
STATES_JA = [
    "昨日届いた注文品が破損していました。レシートは手元にあり、本日中の交換を希望しています。",
    "今月、Pro プランの料金が二重に請求されました。メールを送っても返信がありません。",
    "設定画面を開くたびにアプリが落ちます。Android 14 を使っています。",
    "四半期の売上は 12% 増、費用は 3% 減、人員は据え置きでした。要点をまとめてください。",
    "東京は日本の首都です。富士山は標高 3,776 メートルで、市の西に位置します。",
    "夏目漱石は 1867 年に江戸で生まれ、『吾輩は猫である』で作家としてデビューした。",
]


def _choice(instr, criteria):
    return {"type": "choice", "instructions": instr, "criteria": criteria}


def text_cases():
    cases = []
    q_route = _choice("Which team should handle this request?", ROUTE)
    q_receipt = {"type": "noul", "instructions": "Does the customer have a receipt?"}
    q_urg = {"type": "score", "instructions": "How urgent is this request?", "criteria": ["Routine", "Soon", "Today"]}
    cases.append({"name": "readme_text", "state": STATES_EN[0],
                  "questions": {"route": q_route, "receipt": q_receipt, "urgency": q_urg}})
    for i, s in enumerate(STATES_EN[1:], 1):
        cases.append({"name": "en_%02d" % i, "state": s, "questions": {
            "topic": _choice("What is this text about?", {"billing": "money, charges", "bug": "software faults", "other": None}),
            "neg": {"type": "noul", "instructions": "Is the writer unhappy?"},
            "sev": {"type": "score", "instructions": "How severe is the problem described?",
                    "criteria": ["none", "minor", "moderate", "major", "critical"]}}})
    for i, s in enumerate(STATES_JA):
        cases.append({"name": "ja_%02d" % i, "state": s, "questions": {
            "分類": _choice("この文章はどの分野ですか。", {"請求": "料金や支払い", "不具合": "ソフトウェアの障害", "地理": "場所や地形", "文学": "作家や作品"}),
            "yn": {"type": "noul", "instructions": "この文章は日本に関するものですか。"},
            "緊急度": {"type": "score", "instructions": "緊急度はどれくらいですか。", "criteria": ["低い", "普通", "高い"]}}})
    # noul with custom criteria; JSON state; many options; options without descriptions
    cases.append({"name": "noul_criteria", "state": {"order": 1182, "status": "shipped", "items": [{"sku": "A-1", "qty": 2}]},
                  "questions": {"shipped": {"type": "noul", "instructions": "Has the order shipped?",
                                            "criteria": {"true": "The order has left the warehouse", "false": "The order is still in the warehouse"}}}})
    cases.append({"name": "many_options", "state": "The capital of France is Paris.",
                  "questions": {"letter": _choice("Which city is named?", {c: None for c in
                                ["Paris", "London", "Berlin", "Madrid", "Rome", "Vienna", "Oslo", "Lisbon", "Dublin", "Athens", "Prague", "Warsaw"]})}})
    rng = random.Random(20261011)
    words = "alpha beta gamma delta service invoice router latency customer refund shipping warranty password server login error update".split()
    for i in range(14):
        n = rng.choice([2, 3, 4, 6, 9, 17, 26, 30])
        opts = {"opt%d" % j: ("description of option %d %s" % (j, rng.choice(words)) if rng.random() < .6 else None) for j in range(n)}
        state = " ".join(rng.choice(words) for _ in range(rng.choice([5, 30, 80, 150])))
        qs = {"c": _choice("Pick the best option for the state.", opts),
              "n": {"type": "noul", "instructions": "Is the state about a %s?" % rng.choice(words)},
              "s": {"type": "score", "instructions": "Rate it.", "criteria": ["lvl%d" % k for k in range(rng.choice([2, 5, 10]))] if False else ["lvl%d" % k for k in range(rng.choice([2, 5, 10]))]}}
        cases.append({"name": "rand_%02d" % i, "state": state, "questions": qs})
    return cases


def image_cases():
    qs = {"color": _choice("What is the main colour of the picture?", {"red": None, "blue": None, "green": None, "other": None}),
          "has_circle": {"type": "noul", "instructions": "Does the picture contain a circle?"},
          "bars": {"type": "score", "instructions": "How many distinct objects are visible?", "criteria": ["none", "one", "a few", "many"]}}
    out = []
    for name in ["blue", "circle", "red", "sample_chart"]:
        out.append({"name": "img_" + name, "state": "Look at the attached picture.", "images": ["artifacts/images/%s.png" % name], "questions": qs})
    out.append({"name": "img_chart_tallest", "state": "Look at the attached picture.", "images": ["artifacts/images/sample_chart.png"],
                "questions": {"tallest": _choice("Which bar is the tallest?", {"A": None, "B": None, "C": None, "D": None}),
                              "is_chart": {"type": "noul", "instructions": "Is this a bar chart?"}}})
    out.append({"name": "img_two", "state": "Compare the two pictures.", "images": ["artifacts/images/red.png", "artifacts/images/blue.png"],
                "questions": {"same": {"type": "noul", "instructions": "Do the two pictures have the same main colour?"},
                              "first": _choice("What colour is the first picture?", {"red": None, "blue": None})}})
    out.append({"name": "img_ja", "state": "添付の画像を見てください。", "images": ["artifacts/images/circle.png"],
                "questions": {"circle": {"type": "noul", "instructions": "画像に円がありますか。"},
                              "色": _choice("画像の主な色はどれですか。", {"赤": None, "青": None, "白": None})}})
    return out


def all_cases():
    return text_cases() + image_cases()
