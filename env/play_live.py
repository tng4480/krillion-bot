"""Play a Krillion dive through the game's own UI with a fixed, precomputed guess list (no LLM in the loop).

Opens a fresh Chrome profile (no cookies, so a fresh player), declines optional cookies (Settings -> Save + Exit with
nothing ticked), sets Answer corrections OFF / Clear rejected entries ON / Low-power ON, clicks Begin, then for each
prompt types the list for that prompt as fast as the page takes it: fill the box, click DIVE, wait until the box
empties (rejected) or disappears (landed / time up). Prompts are matched to lists by text. Nothing calls the API
directly and nothing reads answers; the page is driven only through its inputs and buttons.
Writes <out>.json (per-prompt guesses typed, timings, result text) and <out>.png (results screen).
Usage: .venv/bin/python env/play_live.py --lists runs/live/2026-11-20_all/360m.json --out runs/live/2026-11-20_run2
       add --url https://krillion.io/unlimited/classic to test without touching the daily.
"""
import argparse, json, re, time
from pathlib import Path

from playwright.sync_api import sync_playwright

WANT = {"low-power": True, "answer corrections": False, "clear rejected": True}


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def decline_cookies(pg):
    try:
        pg.locator(".cmpboxbtncustom").filter(visible=True).first.click(timeout=8000)
        pg.get_by_text("Save + Exit", exact=True).filter(visible=True).first.click(timeout=5000)
        print("cookies: declined (nothing ticked)")
    except Exception:
        print("cookies: no banner")
    time.sleep(0.5)


def set_settings(pg):
    pg.click("[aria-label='Open settings']")
    time.sleep(0.8)
    for box in pg.locator("input[type=checkbox]").all():
        label = box.evaluate("e => (e.closest('label') || e.parentElement.parentElement.parentElement).innerText")
        lab = label.lower()
        for k, v in WANT.items():
            if k in lab and box.is_checked() != v:
                box.evaluate("e => e.click()") if not box.is_visible() else box.click(force=True)
        print(f"  setting {lab.splitlines()[0][:28]:28} {'ON' if box.is_checked() else 'OFF'}")
    pg.locator("[class*=SettingsControl-module][class*=close]").click(force=True)
    time.sleep(0.5)


def prompt_text(pg):
    m = re.search(r"PROMPT (\d) OF (\d)\s*\n+(.+)", pg.inner_text("body"), re.I)
    return (int(m.group(1)), int(m.group(2)), m.group(3).strip()) if m else (None, None, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", default=None, help="precomputed lists; omit with --model for a real-time run")
    ap.add_argument("--model", default=None, help="real time: read each prompt off the screen and sample now")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--spam", type=int, default=40, help="spam guesses before the 10-answer tail")
    ap.add_argument("--name", default=None, help="short model name from analysis/models.py (sets --model/--adapter)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="https://krillion.io/")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    lists = json.load(open(a.lists)) if a.lists else []
    by_text = {norm(p["text"]): p for p in lists}
    gen = None
    if a.name:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
        from models import MODELS
        a.model, a.adapter = MODELS[a.name]
    if a.model:  # load everything before the dive starts; only sampling happens on the clock
        import sys
        sys.path[:0] = [str(Path(__file__).resolve().parent.parent / d) for d in ("eval", "probe", "data", "env", "analysis")]
        import mlx.core as mx
        from live import build, train_lm
        from play import MLXPolicy
        mx.set_memory_limit(5 * 2**30)
        lm = train_lm()
        pol = MLXPolicy(a.model, a.adapter, temp=1.0, top_p=0.95, max_tokens=480, batch=min(a.n, 64), system="spam")
        pol.sample("Name a fruit", None, None, 1)  # warm-up compile
        gen = lambda text: build(pol.sample(text, None, None, a.n), a.spam, lm=lm)  # noqa: E731
        print("model ready")
    log = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=a.headless)
        pg = b.new_context(viewport={"width": 800, "height": 886}).new_page()
        pg.goto(a.url)
        time.sleep(3)
        decline_cookies(pg)
        set_settings(pg)
        pg.click(".ds-begin-btn", force=True)
        box, dive = pg.locator(".ds-input"), pg.locator(".ds-dive-btn")
        for k in range(len(lists) or 7):
            gen_s, box_wait = None, None
            if gen:  # real time: wait for this prompt's text on screen, sample, then type
                t_q = time.time()
                n, text = None, ""
                while time.time() - t_q < 90:
                    n, total, text = prompt_text(pg)
                    if n == k + 1 and text:
                        break
                    time.sleep(0.1)
                if n != k + 1:
                    break
                t_g = time.time()  # prompt text on screen
                p = dict(pid=f"p{n}", text=text, guesses=gen(text))
                gen_s = round(time.time() - t_g, 1)
                lists.append(p)
            try:
                box.wait_for(state="visible", timeout=90000)
            except Exception:
                break
            t_box = time.time()
            if gen:
                box_wait = round(t_box - t_g, 1)  # prompt on screen -> first keystroke possible
            else:
                n, total, text = prompt_text(pg)
                p = by_text.get(norm(text)) or next((q for t, q in by_text.items()
                                                      if t and norm(text) and (t.startswith(norm(text)) or norm(text).startswith(t))), None)
                if p is None:  # fall back to prompt order (unlimited mode, or a text mismatch)
                    p = lists[(n or k + 1) - 1]
                    print(f"  ! prompt text not in lists, using list {p['pid']}")
            typed = []
            for g in p["guesses"]:
                if not box.is_visible() or not box.is_enabled():
                    break
                try:
                    box.fill(g, timeout=1000)
                    dive.click(timeout=1000, force=True)
                except Exception:
                    break
                typed.append(g)
                t0 = time.time()
                while time.time() - t0 < 3:  # rejected: box empties; landed / time up: box goes away
                    try:
                        if not box.is_visible() or box.input_value(timeout=200) == "":
                            break
                    except Exception:
                        break
                    time.sleep(0.03)
            t_end = time.time()
            desc = pg.get_by_role("button", name=re.compile("DESCEND|SURFACE", re.I))
            try:
                desc.wait_for(state="visible", timeout=40000)
            except Exception:
                pass
            time.sleep(0.6)
            body = pg.inner_text("body")
            m = re.search(r"\n([A-Z][A-Z !'.-]{3,})\n+(.*)\n+([+-]?\d+ PTS[^\n]*)", body)
            res = " | ".join(m.groups()) if m else ""
            rec = dict(n=n, pid=p["pid"], text=text, typed=len(typed), last=typed[-1] if typed else None,
                       seconds=round(t_end - t_box, 1), result=res,
                       sample_s=gen_s, text_to_typing_s=box_wait, guesses=p["guesses"])
            log.append(rec)
            print(f"p{n}: {len(typed)} guesses in {rec['seconds']}s, last '{rec['last']}' -> {res}", flush=True)
            if desc.count() and desc.first.is_visible():
                desc.first.click(force=True)
        try:  # the results screen (score, percentile, catch) loads after Surface
            pg.get_by_text(re.compile("BETTER THAN", re.I)).first.wait_for(timeout=20000)
        except Exception:
            pass
        time.sleep(2)
        body = pg.inner_text("body")
        pg.screenshot(path=a.out + ".png", full_page=True)
        m = re.search(r"COMPLETE\s*\n+(\d+)", body)
        bt = re.search(r"BETTER THAN (\d+)%", body)
        catch = body[body.find("THE CATCH"):body.find("LOGBOOK")] if "THE CATCH" in body else ""
        out = dict(url=a.url, lists=a.lists, score=int(m.group(1)) if m else None,
                   better_than=int(bt.group(1)) if bt else None, prompts=log, catch=catch)
        Path(a.out + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print("score", out["score"], "better than", out["better_than"], "%")
        print(catch)
        b.close()


if __name__ == "__main__":
    main()
