"""Pipeline progress monitor for CriticalNeuroMap.

Wraps run_full_pipeline.py as a subprocess, parses its log output in real
time, and renders a live stage-by-stage progress display.  Zero changes to
any existing pipeline or source file.

Usage (mirrors the pipeline exactly — all args are forwarded):

    python tools/monitor.py                          # run all stages
    python tools/monitor.py --stage dnb_somascan     # single stage
    python tools/monitor.py --stage csd --force      # force rerun
    python tools/monitor.py --stage env_check        # quick smoke test
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Stage catalogue
# ---------------------------------------------------------------------------

STAGE_INFO = {
    "0": ("env_check",       "Environment check"),
    "1": ("adni_preprocess", "ADNI preprocessing          (~30–60 min)"),
    "2": ("ppmi_preprocess", "PPMI preprocessing          (~30–60 min)"),
    "3": ("dnb_somascan",    "DNB SomaScan — PRIMARY      (~1–2 hrs)"),
    "4": ("dnb_olink",       "DNB Olink                   (or skip)"),
    "5": ("cross_platform",  "Cross-platform Golden Set   (~10–20 min)"),
    "6": ("csd",             "CSD analysis — SECONDARY    (~2–4 hrs)"),
    "7": ("validation",      "Validation                  (~20–40 min)"),
    "8": ("ppmi_replication","PPMI replication             (~2–3 hrs)"),
    "9": ("figures",         "Figure generation           (~10–20 min)"),
}

# stage key → stage number (reverse map)
_STAGE_KEY_TO_NUM = {v[0]: k for k, v in STAGE_INFO.items()}

# ---------------------------------------------------------------------------
# Log patterns
# ---------------------------------------------------------------------------

PATTERNS = [
    # Stage transitions
    (re.compile(r"=== Stage (\d+):"), "stage_start"),
    (re.compile(r"Stage '(\S+)' completed in ([\d.]+) seconds"), "stage_done"),
    (re.compile(r"Stage '(\S+)': output exists, skipping"), "stage_skip"),
    # DNB milestones
    (re.compile(r"DNB candidate selection: (\d+)/(\d+) proteins \(top (\d+)%"), "dnb_candidates"),
    (re.compile(r"DNB group identified: (\d+) proteins, score = ([\d.]+)"),      "dnb_group"),
    (re.compile(r"Stage DNB scores computed for (\d+) stages"),                  "dnb_stages_done"),
    (re.compile(r"DNB core proteins: (\d+) proteins appear"),                    "dnb_core"),
    (re.compile(r"DNB on (\w+) complete: (\d+) stages scored, (\d+) core"),      "dnb_platform_done"),
    # Per-participant DNB (no built-in progress — infer from group_identified count)
    (re.compile(r"=== DNB Analysis \((\w+)\) ==="),                              "dnb_platform_start"),
    # CSD milestones
    (re.compile(r"CSD analysis: (\d+)/(\d+) participants have >= (\d+) visits"), "csd_eligible"),
    (re.compile(r"CSD analysis complete: (\d+) participant-protein pairs"),      "csd_pairs_done"),
    (re.compile(r"Sensitivity analysis: window=(\d+), detrending=(\w+)"),        "csd_sens_start"),
    (re.compile(r"Sensitivity analysis complete: (\d+) combinations"),           "csd_sens_done"),
    (re.compile(r"Group CSD statistics: (\d+) proteins tested"),                 "csd_stats"),
    # tqdm lines (CSD inner loop emits these to stderr → merged into stdout)
    (re.compile(r"(\d+)%\|"),                                                    "tqdm"),
    # Pipeline end
    (re.compile(r"Pipeline complete\. Total time: ([\d.]+) seconds"),            "pipeline_done"),
    (re.compile(r"PPMI replication complete"),                                   "ppmi_done"),
]


def _match(line: str) -> tuple[str, re.Match] | None:
    for pat, name in PATTERNS:
        m = pat.search(line)
        if m:
            return name, m
    return None


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _supports_ansi() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") not in ("dumb", "")


ANSI = _supports_ansi()

CLR_LINE = "\033[2K\r"
MOVE_UP  = "\033[{n}A"
BOLD     = "\033[1m"
RESET    = "\033[0m"
GREEN    = "\033[32m"
YELLOW   = "\033[33m"
CYAN     = "\033[36m"
DIM      = "\033[2m"


def _fmt(text: str, *codes: str) -> str:
    if not ANSI:
        return text
    return "".join(codes) + text + RESET


def _elapsed(start: float) -> str:
    secs = int(time.time() - start)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Monitor state
# ---------------------------------------------------------------------------

class MonitorState:
    def __init__(self, single_stage: str | None):
        self.single_stage = single_stage
        self.pipeline_start = time.time()
        # per-stage state: "pending" | "running" | "done" | "skipped"
        self.stage_status: dict[str, str] = {k: "pending" for k in STAGE_INFO}
        self.stage_start_time: dict[str, float] = {}
        self.stage_elapsed: dict[str, float] = {}
        self.current_stage_num: str | None = None
        self.current_detail: str = ""
        self.last_log: str = ""
        self.dnb_groups_found: int = 0
        self.csd_eligible: int | None = None
        self.csd_sens_total: int | None = None
        self.csd_sens_done: int = 0
        self.lock = threading.Lock()
        self.log_path: str = ""

    def on_stage_start(self, num: str) -> None:
        with self.lock:
            self.current_stage_num = num
            self.stage_status[num] = "running"
            self.stage_start_time[num] = time.time()
            self.dnb_groups_found = 0

    def on_stage_done(self, key: str, elapsed_s: float) -> None:
        with self.lock:
            num = _STAGE_KEY_TO_NUM.get(key, self.current_stage_num)
            if num and num in self.stage_status:
                self.stage_status[num] = "done"
                self.stage_elapsed[num] = elapsed_s

    def on_stage_skip(self, key: str) -> None:
        with self.lock:
            num = _STAGE_KEY_TO_NUM.get(key, self.current_stage_num)
            if num and num in self.stage_status:
                self.stage_status[num] = "skipped"

    def update_detail(self, detail: str) -> None:
        with self.lock:
            self.current_detail = detail

    def update_last_log(self, line: str) -> None:
        stripped = line.strip()
        if stripped:
            with self.lock:
                self.last_log = stripped[-120:]  # truncate very long lines


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_DISPLAY_LINES = 0  # how many lines the last render occupied


def render(state: MonitorState) -> None:
    global _DISPLAY_LINES

    with state.lock:
        lines = _build_display(state)

    if ANSI and _DISPLAY_LINES > 0:
        # Move cursor up to overwrite previous render
        sys.stdout.write(MOVE_UP.format(n=_DISPLAY_LINES))

    for line in lines:
        if ANSI:
            sys.stdout.write(CLR_LINE + line + "\n")
        else:
            sys.stdout.write(line + "\n")

    sys.stdout.flush()
    _DISPLAY_LINES = len(lines)


def _build_display(state: MonitorState) -> list[str]:
    SEP = "═" * 62
    DIV = "─" * 62

    lines = []
    lines.append(_fmt("  CriticalNeuroMap Pipeline Monitor", BOLD))
    lines.append(SEP)
    lines.append(
        f"  {'Stage':<6}{'Status':<12}{'Elapsed':<12}Description"
    )
    lines.append(DIV)

    for num, (key, desc) in STAGE_INFO.items():
        status = state.stage_status.get(num, "pending")
        if status == "running":
            sym = _fmt("⟳ running", YELLOW)
            st  = state.stage_start_time.get(num)
            ela = _elapsed(st) if st else "—"
        elif status == "done":
            sym = _fmt("✓ done   ", GREEN)
            ela_s = state.stage_elapsed.get(num)
            ela = _elapsed(time.time() - ela_s + ela_s) if ela_s is None else _fmt_s(ela_s)
        elif status == "skipped":
            sym = _fmt("↷ skip   ", DIM)
            ela = "—"
        else:
            sym = _fmt("– pending", DIM)
            ela = "—"

        # If running a single stage, dim unrelated entries
        if state.single_stage and key != state.single_stage:
            desc_str = _fmt(desc, DIM)
        else:
            desc_str = desc

        lines.append(f"  {num:<6}{sym:<12}{ela:<12}{desc_str}")

    lines.append(SEP)

    # Current activity line
    if state.current_detail:
        lines.append(_fmt(f"  ▶ {state.current_detail}", CYAN))
    else:
        lines.append("")

    # Last meaningful log line
    if state.last_log:
        lines.append(_fmt(f"  ↳ {state.last_log}", DIM))
    else:
        lines.append("")

    # Log file
    elapsed_total = _elapsed(state.pipeline_start)
    lines.append(_fmt(f"  Total: {elapsed_total}  |  log → {state.log_path}", DIM))

    return lines


def _fmt_s(seconds: float) -> str:
    secs = int(seconds)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Log reader thread
# ---------------------------------------------------------------------------

def _reader_thread(proc: subprocess.Popen, state: MonitorState, log_fh, render_event: threading.Event) -> None:
    for raw_line in proc.stdout:
        log_fh.write(raw_line)
        log_fh.flush()
        state.update_last_log(raw_line)
        _handle_line(raw_line, state)
        render_event.set()

    proc.wait()
    render_event.set()  # final render after process ends


def _handle_line(line: str, state: MonitorState) -> None:
    result = _match(line)
    if result is None:
        return

    name, m = result

    if name == "stage_start":
        num = m.group(1)
        state.on_stage_start(num)
        _, desc = STAGE_INFO.get(num, ("?", "Unknown"))
        state.update_detail(f"[Stage {num}] {desc.split('(')[0].strip()}")

    elif name == "stage_done":
        key = m.group(1)
        elapsed_s = float(m.group(2))
        state.on_stage_done(key, elapsed_s)
        state.update_detail(f"Stage '{key}' done in {_fmt_s(elapsed_s)}")

    elif name == "stage_skip":
        key = m.group(1)
        state.on_stage_skip(key)

    elif name == "dnb_platform_start":
        platform = m.group(1)
        state.update_detail(f"[DNB {platform}] Identifying DNB groups per stage…")

    elif name == "dnb_candidates":
        n, total, pct = m.group(1), m.group(2), m.group(3)
        state.update_detail(f"[DNB] Candidate filter: {n}/{total} proteins (top {pct}%)")

    elif name == "dnb_group":
        with state.lock:
            state.dnb_groups_found += 1
            n = state.dnb_groups_found
        state.update_detail(f"[DNB] Groups found so far: {n}  |  last: {m.group(1)} proteins, score={m.group(2)}")

    elif name == "dnb_stages_done":
        state.update_detail(f"[DNB] Stage scores done ({m.group(1)} stages) — running per-participant analysis…")

    elif name == "dnb_core":
        state.update_detail(f"[DNB] Core proteins identified: {m.group(1)}")

    elif name == "dnb_platform_done":
        platform, n_stages, n_core = m.group(1), m.group(2), m.group(3)
        state.update_detail(f"[DNB {platform}] Complete — {n_stages} stages, {n_core} core proteins")

    elif name == "csd_eligible":
        n_elig, n_total, min_v = m.group(1), m.group(2), m.group(3)
        with state.lock:
            state.csd_eligible = int(n_elig)
        state.update_detail(f"[CSD] {n_elig}/{n_total} participants eligible (≥{min_v} visits) — rolling window in progress…")

    elif name == "tqdm":
        pct = m.group(1)
        with state.lock:
            elig = state.csd_eligible
        desc = f"participants" if elig is None else f"{elig} participants"
        state.update_detail(f"[CSD] Rolling-window progress: {pct}% of {desc}")

    elif name == "csd_pairs_done":
        state.update_detail(f"[CSD] Rolling-window complete — {m.group(1)} participant-protein pairs")

    elif name == "csd_sens_start":
        window, method = m.group(1), m.group(2)
        with state.lock:
            state.csd_sens_done += 1
            done = state.csd_sens_done
        state.update_detail(f"[CSD sensitivity] Combo {done}: window={window}, detrending={method}")

    elif name == "csd_sens_done":
        state.update_detail(f"[CSD sensitivity] Done — {m.group(1)} combinations complete")

    elif name == "csd_stats":
        state.update_detail(f"[CSD] Group statistics: {m.group(1)} proteins tested")

    elif name == "ppmi_done":
        state.update_detail("[PPMI] Replication complete")

    elif name == "pipeline_done":
        total = _fmt_s(float(m.group(1)))
        state.update_detail(f"Pipeline complete in {total}")


# ---------------------------------------------------------------------------
# Watch mode: tail an existing log file written by the pipeline
# ---------------------------------------------------------------------------

def _tail_reader_thread(log_file: Path, state: MonitorState, render_event: threading.Event, stop_event: threading.Event) -> None:
    """Follow a log file as it grows, feeding lines to the state machine."""
    with open(log_file, "r") as fh:
        # Replay lines already in the file before we started watching
        for line in fh:
            state.update_last_log(line)
            _handle_line(line, state)

        # Then follow new lines as they arrive
        while not stop_event.is_set():
            line = fh.readline()
            if line:
                state.update_last_log(line)
                _handle_line(line, state)
                render_event.set()
            else:
                time.sleep(0.2)


def _run_watch_mode(log_file: Path, project_root: Path) -> None:
    """Watch an existing log file and render progress from it."""
    if not log_file.exists():
        print(f"Waiting for log file: {log_file}", flush=True)
        while not log_file.exists():
            time.sleep(0.5)

    state = MonitorState(single_stage=None)
    try:
        state.log_path = str(log_file.relative_to(project_root))
    except ValueError:
        state.log_path = str(log_file)

    render_event = threading.Event()
    stop_event = threading.Event()

    def _on_sigint(sig, frame):
        stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)

    reader = threading.Thread(
        target=_tail_reader_thread,
        args=(log_file, state, render_event, stop_event),
        daemon=True,
    )
    reader.start()

    try:
        while not stop_event.is_set():
            render_event.wait(timeout=5)
            render_event.clear()
            render(state)
            # Stop automatically once the pipeline signals completion
            with state.lock:
                done = "Pipeline complete" in state.current_detail
            if done:
                time.sleep(1)
                render(state)
                break
    finally:
        stop_event.set()

    if ANSI:
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    pipeline_script = project_root / "pipelines" / "run_full_pipeline.py"

    raw_args = sys.argv[1:]

    # --watch <logfile>  →  tail mode; don't launch the pipeline
    if raw_args and raw_args[0] == "--watch":
        if len(raw_args) < 2:
            # Default to logs/pipeline_run.log if no file given
            log_file = project_root / "logs" / "pipeline_run.log"
        else:
            log_file = Path(raw_args[1])
            if not log_file.is_absolute():
                log_file = project_root / log_file
        _run_watch_mode(log_file, project_root)
        return

    # ------------------------------------------------------------------
    # Subprocess mode: launch the pipeline and capture its output
    # ------------------------------------------------------------------
    args = raw_args

    # Detect --stage value for display filtering
    single_stage = None
    for i, a in enumerate(args):
        if a == "--stage" and i + 1 < len(args):
            single_stage = args[i + 1]
            break

    # Log file in project root
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = project_root / f"pipeline_run_{ts}.log"

    state = MonitorState(single_stage)
    state.log_path = str(log_path.relative_to(project_root))

    # If only running a single stage, mark others as skipped upfront
    if single_stage:
        for num, (key, _) in STAGE_INFO.items():
            if key != single_stage:
                state.stage_status[num] = "skipped"

    cmd = [sys.executable, str(pipeline_script)] + args
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    render_event = threading.Event()

    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(project_root),
            )

            def _on_sigint(sig, frame):
                proc.send_signal(signal.SIGINT)
                sys.exit(130)

            signal.signal(signal.SIGINT, _on_sigint)

            reader = threading.Thread(
                target=_reader_thread,
                args=(proc, state, log_fh, render_event),
                daemon=True,
            )
            reader.start()

            # Main loop: redraw whenever the reader signals an update
            while reader.is_alive() or render_event.is_set():
                render_event.wait(timeout=5)  # also redraw every 5s for elapsed time
                render_event.clear()
                render(state)

            reader.join()
            render(state)  # final render

        return_code = proc.returncode

    except FileNotFoundError:
        print(f"Error: pipeline script not found at {pipeline_script}", file=sys.stderr)
        sys.exit(1)

    if ANSI:
        print()  # blank line after final render
    print(f"Full log saved to: {log_path}")
    sys.exit(return_code if return_code is not None else 0)


if __name__ == "__main__":
    main()
