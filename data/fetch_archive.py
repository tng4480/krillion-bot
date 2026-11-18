"""Fetch past dives' answer lists from krillionanswer.today into data/reveal/<date>.json.

The site (robots.txt: Allow /) has a page per dive with its 7 prompt texts, and one static JSON per prompt at
/answers/<dive>/<pid>.json with every answer's tier and score. That's ~98% of the game's own reveal (checked on
Dive 126 p3: 658 of 669 answers, tiers agree) but without aliases, so day captures from the game itself
(reveal files without "source") are kept and never overwritten.

Polite: one request per --delay seconds, resumable (skips dates already on disk).
Usage: .venv/bin/python data/fetch_archive.py [--first 1] [--last 126]
"""
import argparse, datetime, json, re, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "https://krillionanswer.today"
UA = {"User-Agent": "Mozilla/5.0 (research; small polite crawler)"}


def get(url, delay):
    time.sleep(delay)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # transient errors: back off and retry
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def parse_dive(html):
    t = re.search(r"<title>Krillion (\w+ \d+, \d{4}) \| Dive (\d+) Answers", html)
    date = datetime.datetime.strptime(t.group(1), "%B %d, %Y").date().isoformat()
    # the prompt JSON sits inside an escaped JS string: \" is a JSON quote, \\\" a quote inside a prompt text
    flat = html.replace('\\\\\\"', '\x00').replace('\\"', '"').replace('\x00', '\\"')
    prompts, seen = [], set()
    for m in re.finditer(r'"id":"(p\d+)","text":"((?:[^"\\]|\\.)*)","category":"([^"]*)"', flat):
        if m.group(1) in seen:
            continue
        seen.add(m.group(1))
        prompts.append(dict(id=m.group(1), text=json.loads('"' + m.group(2) + '"'), site_category=m.group(3)))
    return date, int(t.group(2)), prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=126)
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()
    out_dir = ROOT / "reveal"
    for n in range(a.first, a.last + 1):
        try:
            date, dive, prompts = parse_dive(get(f"{BASE}/krillion-answers-archive/krillion-dive-{n}/", a.delay))
        except Exception as e:
            print(f"dive {n}: page failed ({e})", flush=True); continue
        path = out_dir / f"{date}.json"
        if path.exists():
            print(f"dive {n} {date}: already on disk, skipped", flush=True); continue
        if len(prompts) != 7:
            print(f"dive {n} {date}: found {len(prompts)} prompts (expected 7)", flush=True)
        for p in prompts:
            try:
                p["answers"] = json.loads(get(f"{BASE}/answers/{n}/{p['id']}.json", a.delay))
            except Exception as e:
                print(f"dive {n} {p['id']}: answers failed ({e})", flush=True); p["answers"] = []
        prompts = [p for p in prompts if p["answers"]]
        if prompts:
            path.write_text(json.dumps(dict(date=date, dive=dive, source="krillionanswer.today", prompts=prompts),
                                       ensure_ascii=False))
        print(f"dive {n} {date}: " + " ".join(f"{p['id']}={len(p['answers'])}" for p in prompts), flush=True)


if __name__ == "__main__":
    main()
