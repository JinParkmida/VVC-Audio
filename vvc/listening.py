"""Blind listening test: can a human understand VVC audio where Whisper can't (or vice versa)?

Stimuli come from tools/make_listening_test.py. This screen reads only out/listening/playlist.json
(clip ids and audio files). The sealed answer key is opened after the last answer, and every clip
is checked against its SHA-256 checksum.
Run with  python -m vvc listen   or press  l  in the main VVC UI.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

from .eval_text import words_correct

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "out" / "listening"
PLAYLIST = DIR / "playlist.json"

try:
    import winsound
except ImportError:
    winsound = None

CONF_LABELS = {1: "1 · guessing", 2: "2", 3: "3 · fairly sure", 4: "4", 5: "5 · certain"}

INTRO = """[b]Blind listening test[/b]

You will hear [b]{n} short Danish sentences[/b], one at a time. Each one has been squeezed through
VVC to a different degree, from untouched to almost destroyed. You don't know which, and every
sentence is new to you.

For each clip:
  1. It plays automatically. [b]Ctrl+R[/b] replays it (replays are counted, use them as you like).
  2. Type [b]what you heard[/b], as much as you can. Write single words if that's all you caught. Leave it
     empty if you heard nothing. Spelling doesn't need to be perfect. Press [b]Enter[/b].
  3. Press [b]1–5[/b] for how sure you are: 1 = guessing, 5 = certain.

There's no going back, and answers are saved after every clip. At the end you see how you did
against Whisper on exactly the same audio, condition by condition.
Use headphones if you can. About 10 minutes."""


def load_playlist() -> dict | None:
    if not PLAYLIST.exists():
        return None
    return json.loads(PLAYLIST.read_text(encoding="utf-8"))


def load_key(playlist: dict) -> dict:
    """Open the sealed answer key, and check that every clip is the archived one."""
    key = json.loads((ROOT / playlist["key"]).read_text(encoding="utf-8"))
    bad = [t["id"] for t in key["trials"]
           if hashlib.sha256((ROOT / t["file"]).read_bytes()).hexdigest() != t["sha256_stimulus"]]
    key["verified"] = not bad
    key["modified_clips"] = bad
    return key


def score(manifest: dict, answers: list[dict]) -> dict:
    """Per-condition human vs Whisper comparison."""
    by_id = {t["id"]: t for t in manifest["trials"]}
    rows = []
    for a in answers:
        t = by_id[a["id"]]
        rows.append({**t, **a,
                     "human_correct": words_correct(t["sentence"], a["answer"]),
                     "whisper_correct": words_correct(t["sentence"], t["whisper"])})
    conds = []
    for c, desc in manifest["conditions"].items():
        g = [r for r in rows if r["condition"] == c]
        if not g:
            continue
        conds.append(dict(condition=c, description=desc, n=len(g),
                          human=float(np.mean([r["human_correct"] for r in g])),
                          whisper=float(np.mean([r["whisper_correct"] for r in g])),
                          human_conf=float(np.mean([r["confidence"] for r in g])),
                          whisper_conf=float(np.mean([r["whisper_confidence"] or 0 for r in g])),
                          plays=float(np.mean([r["plays"] for r in g]))))
    wrong = lambda r, who: r[f"{who}_correct"] < 0.5
    return dict(
        rows=rows, conditions=conds,
        human_total=float(np.mean([r["human_correct"] for r in rows])) if rows else 0.0,
        whisper_total=float(np.mean([r["whisper_correct"] for r in rows])) if rows else 0.0,
        human_confidently_wrong=[r for r in rows if wrong(r, "human") and r["confidence"] >= 4],
        whisper_confidently_wrong=[r for r in rows if wrong(r, "whisper") and (r["whisper_confidence"] or 0) >= 0.5],
        human_only=[r for r in rows if r["human_correct"] - r["whisper_correct"] >= 0.5],
        whisper_only=[r for r in rows if r["whisper_correct"] - r["human_correct"] >= 0.5],
    )


def report_md(manifest: dict, s: dict, when: str) -> str:
    L = [f"# Listening test — {when}", "",
         f"{len(s['rows'])} trials, Danish, voice {manifest['voice']}, run {manifest['run_id']}, "
         f"recognizer {manifest['recognizer']}. *Words correct* uses lenient matching (small typos "
         f"count as correct) for both you and Whisper. Clip integrity: "
         f"{'all clips match their sealed SHA-256 checksums' if manifest.get('verified') else 'MODIFIED: ' + ', '.join(manifest.get('modified_clips', []))}.", "",
         f"**Overall words correct: you {s['human_total']:.0%}, Whisper {s['whisper_total']:.0%}**", "",
         "| Condition | n | You: words correct | Whisper: words correct | Your confidence (1–5) | "
         "Whisper confidence | Your plays |", "|---|---|---|---|---|---|---|"]
    for c in s["conditions"]:
        L.append(f"| {c['description']} | {c['n']} | {c['human']:.0%} | {c['whisper']:.0%} | "
                 f"{c['human_conf']:.1f} | {c['whisper_conf']:.0%} | {c['plays']:.1f} |")
    def lst(title, rs):
        out = ["", f"## {title} ({len(rs)})", ""]
        for r in rs:
            out += [f"- *{r['description']}* — \"{r['sentence']}\"",
                    f"  - you ({r['confidence']}/5): {r['answer'] or '(nothing)'}",
                    f"  - Whisper ({(r['whisper_confidence'] or 0):.0%}): {r['whisper'] or '(nothing)'}"]
        return out
    L += lst("You understood, Whisper didn't", s["human_only"])
    L += lst("Whisper understood, you didn't", s["whisper_only"])
    L += lst("You were confidently wrong (≥ 4/5, < 50 % words)", s["human_confidently_wrong"])
    L += lst("Whisper was confidently wrong (≥ 50 % confidence, < 50 % words)", s["whisper_confidently_wrong"])
    L += ["", "## All trials", "", "| # | Condition | Sentence | You | Whisper |", "|---|---|---|---|---|"]
    for i, r in enumerate(s["rows"], 1):
        L.append(f"| {i} | {r['description']} | {r['sentence']} | {r['answer'] or '—'} "
                 f"({r['human_correct']:.0%}) | {r['whisper'] or '—'} ({r['whisper_correct']:.0%}) |")
    return "\n".join(L)


class ListeningScreen(Screen):
    DEFAULT_CSS = """
    ListeningScreen { align: center top; }
    #lt-body { width: 100; max-width: 100%; height: 1fr; padding: 1 2; }
    #lt-intro { margin-bottom: 1; }
    #lt-progress { color: $accent; text-style: bold; margin-bottom: 1; }
    #lt-status { color: $text-muted; height: 2; }
    #lt-answer { margin: 1 0; }
    #lt-conf { height: auto; margin-top: 1; }
    #lt-conf Button { width: 1fr; min-width: 10; }
    #lt-conf-label { margin-top: 1; color: $text-muted; }
    #lt-actions { height: auto; margin-top: 1; }
    #lt-summary { height: auto; margin-bottom: 1; }
    #lt-table { height: auto; max-height: 20; }
    .hidden { display: none; }
    """
    BINDINGS = [
        Binding("ctrl+r", "replay", "Replay clip", priority=True),
        *[Binding(str(k), f"confidence({k})", f"sure: {k}", show=(k in (1, 5))) for k in range(1, 6)],
        Binding("escape", "leave", "Back"),
    ]

    def __init__(self):
        super().__init__()
        self.playlist = load_playlist()
        self.manifest = None                     # the answer key, opened only in finish()
        self.i = -1
        self.answers: list[dict] = []
        self.plays = 0
        self.t_start = 0.0
        self.pending_answer: str | None = None
        self.stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="lt-body"):
            n = len(self.playlist["clips"]) if self.playlist else 0
            yield Static(INTRO.format(n=n) if self.playlist else
                         "[red]No listening test found.[/] Create one first:\n\n"
                         "    python tools/make_listening_test.py", id="lt-intro")
            yield Label("", id="lt-progress", classes="hidden")
            yield Static("", id="lt-status")
            yield Input(placeholder="type what you heard, then Enter", id="lt-answer", classes="hidden")
            yield Label("How sure are you? Press 1–5.", id="lt-conf-label", classes="hidden")
            with Horizontal(id="lt-conf", classes="hidden"):
                for k, lab in CONF_LABELS.items():
                    yield Button(lab, id=f"conf-{k}")
            with Horizontal(id="lt-actions"):
                if self.playlist:
                    yield Button("Start test", id="lt-start", variant="primary")
                yield Button("▶ Replay  (Ctrl+R)", id="lt-replay", classes="hidden")
                yield Button("Back", id="lt-back")
            yield Static("", id="lt-summary", classes="hidden")
            yield DataTable(id="lt-table", classes="hidden", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        if self.playlist:
            self.query_one("#lt-start", Button).focus()

    # ------------------------------------------------------------- helpers
    def show(self, sel: str, on_: bool = True) -> None:
        self.query_one(sel).set_class(not on_, "hidden")

    def play(self) -> None:
        t = self.playlist["clips"][self.i]
        if winsound is not None:
            winsound.PlaySound(str(ROOT / t["file"]), winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.plays += 1
        self.query_one("#lt-status", Static).update(
            f"♪ playing… (heard {self.plays}×)" if winsound else "[red]Playback needs Windows (winsound).[/]")

    def save(self) -> None:
        DIR.mkdir(parents=True, exist_ok=True)
        (DIR / f"answers_{self.stamp}.json").write_text(json.dumps(
            {"run_id": self.playlist["run_id"], "answers": self.answers}, indent=1, ensure_ascii=False),
            encoding="utf-8")

    # ------------------------------------------------------------- flow
    @on(Button.Pressed, "#lt-start")
    def start(self) -> None:
        for sel in ("#lt-intro", "#lt-start"):
            self.show(sel, False)
        for sel in ("#lt-progress", "#lt-answer", "#lt-replay"):
            self.show(sel)
        self.next_trial()

    def next_trial(self) -> None:
        self.i += 1
        trials = self.playlist["clips"]
        if self.i >= len(trials):
            self.finish()
            return
        self.plays, self.pending_answer, self.t_start = 0, None, time.time()
        self.query_one("#lt-progress", Label).update(f"Clip {self.i + 1} of {len(trials)}")
        inp = self.query_one("#lt-answer", Input)
        inp.value, inp.disabled = "", False
        self.show("#lt-conf", False)
        self.show("#lt-conf-label", False)
        inp.focus()
        self.play()

    @on(Input.Submitted, "#lt-answer")
    def answered(self, event: Input.Submitted) -> None:
        self.pending_answer = event.value.strip()
        event.input.disabled = True
        self.show("#lt-conf")
        self.show("#lt-conf-label")
        self.query_one("#conf-3", Button).focus()

    @on(Button.Pressed, "#lt-conf Button")
    def conf_button(self, event: Button.Pressed) -> None:
        self.action_confidence(int(event.button.id.split("-")[1]))

    def action_confidence(self, k: int) -> None:
        if self.pending_answer is None or self.i < 0:
            return
        t = self.playlist["clips"][self.i]
        self.answers.append(dict(id=t["id"], answer=self.pending_answer, confidence=k,
                                 plays=self.plays, seconds=round(time.time() - self.t_start, 1)))
        self.save()
        self.next_trial()

    def action_replay(self) -> None:
        if self.playlist and 0 <= self.i < len(self.playlist["clips"]):
            self.play()

    @on(Button.Pressed, "#lt-replay")
    def replay_button(self) -> None:
        self.action_replay()
        if self.pending_answer is None:
            self.query_one("#lt-answer", Input).focus()

    @on(Button.Pressed, "#lt-back")
    def back_button(self) -> None:
        self.action_leave()

    def action_leave(self) -> None:
        if winsound is not None:
            winsound.PlaySound(None, 0)
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        else:
            self.app.exit()

    def finish(self) -> None:
        if winsound is not None:
            winsound.PlaySound(None, 0)
        for sel in ("#lt-answer", "#lt-conf", "#lt-conf-label", "#lt-replay"):
            self.show(sel, False)
        self.manifest = load_key(self.playlist)
        s = score(self.manifest, self.answers)
        md = DIR / f"results_{self.stamp}.md"
        md.write_text(report_md(self.manifest, s, self.stamp), encoding="utf-8")
        (DIR / f"results_{self.stamp}.json").write_text(json.dumps(
            {k: v for k, v in s.items()}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        self.query_one("#lt-progress", Label).update("Done. Thank you!")
        lines = [f"[b]Words correct overall — you: {s['human_total']:.0%}   Whisper: {s['whisper_total']:.0%}[/]", ""]
        if s["human_only"]:
            lines.append(f"You understood [b]{len(s['human_only'])}[/] clip(s) that Whisper didn't.")
        if s["whisper_only"]:
            lines.append(f"Whisper understood [b]{len(s['whisper_only'])}[/] clip(s) that you didn't.")
        lines.append(f"Confidently wrong: you {len(s['human_confidently_wrong'])}×, "
                     f"Whisper {len(s['whisper_confidently_wrong'])}×.")
        lines.append(f"\nFull report with every sentence: [b]{md.relative_to(ROOT) if md.is_relative_to(ROOT) else md}[/]")
        self.query_one("#lt-status", Static).update("")
        summary = self.query_one("#lt-summary", Static)
        summary.update("\n".join(lines))
        self.show("#lt-summary")
        table = self.query_one("#lt-table", DataTable)
        table.add_columns("Condition", "n", "You", "Whisper", "Your sureness", "Whisper conf.")
        for c in s["conditions"]:
            table.add_row(c["description"], str(c["n"]), f"{c['human']:.0%}", f"{c['whisper']:.0%}",
                          f"{c['human_conf']:.1f}/5", f"{c['whisper_conf']:.0%}")
        self.show("#lt-table")
        self.query_one("#lt-back", Button).focus()


class ListeningApp(App):
    TITLE = "VVC — listening test"

    def get_default_screen(self) -> Screen:
        return ListeningScreen()


def main() -> None:
    ListeningApp().run()
