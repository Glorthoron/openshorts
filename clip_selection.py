"""
Pure helpers for the Gemini clip-selection pipeline.

Standard-library only so both main.py and gemini_worker.py can import it and
the logic stays unit-testable without the heavy video dependencies.
"""
import math
import re

# USD per 1M tokens (input, output incl. thinking), from ai.google.dev pricing.
MODEL_PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),  # deprecated (shut down 2026-06-01)
}


def lookup_model_prices(model_name):
    """Longest-prefix match against MODEL_PRICES; None if unknown."""
    name = str(model_name or "").lower()
    best_key = None
    for key in MODEL_PRICES:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return MODEL_PRICES[best_key] if best_key else None


def clip_count_targets(n_windows):
    """How many clips to ask the detail pass for, given the shortlist size.

    Measured on prod 3-ago-2026: 408 of 429 jobs (95%) delivered 3 clips or
    fewer, the mode being ONE, while the prompt was free to return one per
    shortlisted window. Users who received 1-3 clips came back a second day
    0.4% of the time; those who received 4-9 came back 16.1% — so the clip
    count, not the clip quality, is what the retention curve hangs on.

    The old prompt biased hard the other way ("prefer one great clip per
    candidate window") and handed the model two unbounded licences to drop
    clips (the 2-second rule and STANDS ALONE both end in "or skip it"), with
    no floor to stop it collapsing to a single clip. This puts a floor and a
    realistic ceiling on it instead.

    ``CLIP_TARGET_MIN`` / ``CLIP_TARGET_MAX`` override both for A/B runs
    without a deploy (the reframe-testing harness drives them).
    """
    import os

    n = max(1, int(n_windows or 1))
    # Floor grows with the material: 3 windows -> 3, 5 -> 4, 10+ -> 6.
    low = max(2, min(6, n // 2 + 2))
    # Ceiling allows a rich window to yield more than one without inviting padding.
    high = min(12, max(4, n * 2))
    low = min(low, high)

    def _override(name, current):
        raw = os.environ.get(name)
        if not raw:
            return current
        try:
            return max(1, int(raw))
        except ValueError:
            return current

    low = _override("CLIP_TARGET_MIN", low)
    high = _override("CLIP_TARGET_MAX", high)
    return low, max(low, high)


# --- A long source gives clips in proportion to its length -----------------
# The count used to stop growing at twelve minutes of source: the shortlist
# was capped at 10 windows, so a five-hour video came back with the same 6-12
# clips as a quarter of an hour, while the minutes charged for it grew with
# every minute of it.

DETAIL_GROUP = 10            # windows one detail call reads


def _env_number(name, default):
    import os
    try:
        return max(0.0, float(os.environ.get(name, "") or default))
    except ValueError:
        return float(default)


def max_clips_per_job():
    """The most clips one job delivers (``CLIP_MAX_PER_JOB``, 60)."""
    return max(1, int(_env_number("CLIP_MAX_PER_JOB", 60)))


def targets_pinned():
    """Whether the count was set from outside (``CLIP_TARGET_MIN`` / ``MAX``:
    the user's own number, or the free plan's cap): then nothing here adds to it."""
    import os
    return bool(os.environ.get("CLIP_TARGET_MIN") or os.environ.get("CLIP_TARGET_MAX"))


def length_floor(video_duration):
    """The clips a source this long gives at least: one per
    ``CLIP_LONG_SECONDS_PER_CLIP`` (300, five minutes; 0 turns it off), up to
    ``max_clips_per_job``. 0 while the fixed band already covers the source
    (half an hour is six clips either way) or when the count is pinned."""
    try:
        seconds = float(video_duration or 0)
    except (TypeError, ValueError):
        return 0
    per = _env_number("CLIP_LONG_SECONDS_PER_CLIP", 300)
    if per <= 0 or targets_pinned():
        return 0
    n = int(seconds // per)
    return min(max_clips_per_job(), n) if n > 6 else 0


def shortlist_target(video_duration):
    """How many scored windows the detail pass should get.

    Scales with duration, floored at 3 so a short source still gets a choice.
    Up to half an hour it stops at 10. Past that it follows ``length_floor``:
    five windows for every three clips of the floor, which is the share of a
    group of ten the detail pass is asked to turn into clips
    (``clip_count_targets``), so the floor is reachable without making every
    shortlisted window yield one. ``score_batches`` exists to make this number
    reachable.
    """
    try:
        seconds = float(video_duration or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    base = max(3, min(10, int(seconds // 90) + 2))
    return max(base, math.ceil(length_floor(seconds) * 5 / 3))


def detail_groups(shortlist):
    """The shortlist in the groups the detail pass reads, one call each.

    A shortlist one call can read (``DETAIL_GROUP`` windows) is one group, as
    it always was and in the order it came. A longer one (a long source) goes
    in video order in near-equal groups: each call then covers one stretch of
    the video, asked for its own 6-12 clips, so the clips come from all of it
    and the total follows the length."""
    shortlist = list(shortlist or [])
    if len(shortlist) <= DETAIL_GROUP:
        return [shortlist] if shortlist else []

    def start(w):
        try:
            return float(w.get("start") or 0)
        except (AttributeError, TypeError, ValueError):
            return 0.0
    return score_batches(sorted(shortlist, key=start), DETAIL_GROUP)


def detail_plan(shortlist):
    """``(groups, bands, min_clips, max_clips)`` for the detail pass: each
    group with its own ``clip_count_targets`` band, and the job's totals. One
    group: exactly its band. Several: the sum, the ceiling never past
    ``max_clips_per_job``."""
    groups = detail_groups(shortlist)
    if not groups:
        low, high = clip_count_targets(0)
        return [], [], low, high
    bands = [clip_count_targets(len(g)) for g in groups]
    if len(groups) == 1:
        return groups, bands, bands[0][0], bands[0][1]
    cap = max_clips_per_job()
    low = min(cap, sum(b[0] for b in bands))
    high = max(low, min(cap, sum(b[1] for b in bands)))
    return groups, bands, low, high


def vision_targets(video_duration, n_segments, low, high):
    """``(min per stretch, max per stretch, job minimum, job maximum)`` for the
    vision pass (footage with no speech), from its classic ``low``-``high``.

    It has no scoring windows, so a long source's count comes straight from
    ``length_floor``: at least that many clips, at most twice (never past
    ``max_clips_per_job``), spread over the stretches the model watches.
    Without a floor (a short source, a pinned count) it is what it was: every
    stretch may return up to ``high`` and the job keeps the best ``high``."""
    n = max(1, int(n_segments or 1))
    floor = length_floor(video_duration)
    if not floor:
        return max(1, math.ceil(low / n)), high, low, high
    job_low = max(low, floor)
    job_high = max(high, job_low, min(max_clips_per_job(), 2 * floor))
    per_low = max(1, math.ceil(job_low / n))
    return per_low, max(per_low, math.ceil(job_high / n)), job_low, job_high


def score_batches(windows, batch_size):
    """Split ``windows`` into near-equal scoring batches.

    Two things were wrong with the old ``range(0, n, batch_size)`` walk plus a
    prompt that said "choose up to 3 windows from this batch".

    The cap made the shortlist unreachable. Its real ceiling was
    ``3 * n_batches``, not ``shortlist_target``: measured 22-sep-2026 on a
    9:10 source, 9 windows batched 8 + 1 returned 3 + 1 = 4 windows against a
    target of 8. ``clip_count_targets`` then derives the clip floor from that
    halved shortlist, so the detail pass was asked for 4-8 clips instead of
    6-12 and the job delivered 4 (3 in prod on the same video). Every source
    under ~30 minutes was starved the same way, which is the mechanism behind
    "95% of jobs deliver 3 clips or fewer" — the clip count the retention
    curve hangs on (see clip_count_targets). The prompt now scores every
    window it is given and the shortlist is the global top ``target``, so the
    selection happens where the scores can actually be compared.

    Per-batch selection also threw away the ranking it was computing: a batch
    holding five great moments could only contribute three, while a batch of
    filler still contributed its own three.

    The remainder was its own bug: a trailing batch of one window came back
    with that window whatever its score, because a cap of 3 cannot filter a
    batch of 1 — the tail of the video entered the shortlist by arithmetic
    rather than merit (``window_009`` did exactly that on the measured run).
    Near-equal batches keep that from happening and also keep each score
    comparable, since a window judged alone is judged against nothing.
    """
    windows = list(windows or [])
    n = len(windows)
    if not n:
        return []
    size = max(1, int(batch_size or 1))
    n_batches = max(1, -(-n // size))       # ceil
    base, extra = divmod(n, n_batches)

    out = []
    start = 0
    for index in range(n_batches):
        take = base + (1 if index < extra else 0)
        batch = windows[start:start + take]
        start += take
        if batch:
            out.append(batch)
    return out


def trim_to_best(shorts, max_clips):
    """Cut an over-long detail-pass result down to ``max_clips`` BY SCORE.

    The detail pass hands its clips back in transcript order, batch after
    batch, so slicing the list keeps the EARLIEST clips rather than the best
    ones. On a 9-minute walkthrough that quietly threw away everything past
    minute three: the model proposed clips across the whole video, and the
    ones covering the demo, the MCP walkthrough and the close were the tail
    that got dropped. Worse, the failure scales the wrong way — the more
    generous the model is, the more of the video disappears.

    That sabotages the windowing: get_viral_clips builds scoring windows
    precisely because "a single call over the whole transcript clusters picks
    near the start", and a positional slice puts the clustering right back.

    Ranking is by ``predicted_score`` (the detail prompt already asks for it,
    and nothing else was reading it here). Ties keep transcript order, and the
    survivors come back in transcript order too, so clip numbering still runs
    front to back the way every caller downstream expects.
    """
    max_clips = max(1, int(max_clips or 1))
    if len(shorts) <= max_clips:
        return list(shorts)

    def score(item):
        try:
            return float(item[1].get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    indexed = list(enumerate(shorts))
    best = sorted(indexed, key=score, reverse=True)[:max_clips]
    return [item for _, item in sorted(best, key=lambda pair: pair[0])]


def vision_segments(video_duration, segment_seconds, max_segments):
    """Split a silent source into near-equal ``(start, end)`` stretches for
    the vision pass, each at most ``segment_seconds`` long.

    The vision pass sends Gemini the footage itself at ~300 tokens/second, so
    an hour (~1.08M tokens) no longer fits a 1M-token window and a silent
    multi-hour capture failed at the model. Near-equal stretches, not fixed
    ones: a 41-minute source split at 40 minutes would leave a one-minute
    tail that the prompt has to fill with 3+ clips. A source that fits in one
    stretch returns one stretch, so short videos take the single-upload path
    unchanged. Past ``max_segments`` the source is truncated (the caller logs
    it) rather than spending without bound on one job.
    """
    duration = max(0.0, float(video_duration or 0))
    seg = max(60.0, float(segment_seconds))
    if duration <= seg:
        return [(0.0, round(duration, 3))]
    n = int(math.ceil(duration / seg))
    n_kept = max(1, min(n, int(max_segments)))
    step = duration / n
    return [(round(i * step, 3), round(min(duration, (i + 1) * step), 3))
            for i in range(n_kept)]


def clip_duration_bounds():
    """The clip length band (seconds) the selection prompts and word-snapping
    enforce. ``CLIP_MIN_SECONDS`` / ``CLIP_MAX_SECONDS`` override the classic
    15-60 — set per job by /api/process when the user asks for a specific
    length, or by hand for A/B runs. Values are clamped to platform-sane
    limits and re-ordered so bad input degrades instead of breaking the job.
    """
    import os

    def _read(name, default):
        try:
            return float(os.environ.get(name, ""))
        except ValueError:
            return default

    lo = _read("CLIP_MIN_SECONDS", 15.0)
    hi = _read("CLIP_MAX_SECONDS", 60.0)
    lo = min(max(lo, 5.0), 175.0)
    hi = min(max(hi, 10.0), 180.0)
    if hi < lo + 5.0:  # keep a real band: degenerate ranges starve the model
        hi = min(180.0, lo + 5.0)
    return round(lo, 3), round(hi, 3)


def compact_words(words, precision=2):
    """Round word timestamps for prompts — full float precision wastes tokens."""
    return [
        {
            "w": w.get("w", ""),
            "s": round(float(w.get("s", 0)), precision),
            "e": round(float(w.get("e", 0)), precision),
        }
        for w in words
    ]


def build_transcript_windows(transcript_result, video_duration,
                             window_seconds=90, overlap_seconds=30):
    """
    Build scoring windows aligned to Whisper segment boundaries, so a sentence
    (and usually a viral moment) is never cut in half mid-window. Windows grow
    segment by segment to roughly window_seconds (up to 1.25x for the closing
    segment) and the next window starts ~overlap_seconds before the previous
    end, also snapped to a segment start.
    """
    segments = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((float(segment.get("start", 0)), float(segment.get("end", 0)), text))

    windows = []
    window_index = 1
    i = 0
    n = len(segments)
    while i < n:
        w_start = segments[i][0]
        j = i
        # Extend while the NEXT segment still fits within a tolerant cap, so the
        # window closes on a segment boundary near window_seconds.
        while j + 1 < n and segments[j + 1][1] - w_start <= window_seconds * 1.25:
            j += 1
            if segments[j][1] - w_start >= window_seconds:
                break
        w_end = segments[j][1]
        windows.append({
            "id": f"window_{window_index:03d}",
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "text": " ".join(seg[2] for seg in segments[i:j + 1]),
        })
        window_index += 1

        if j >= n - 1:
            break
        # Next window starts at the first segment beginning after (end - overlap),
        # but always makes progress.
        target = w_end - overlap_seconds
        k = i + 1
        while k <= j and segments[k][0] < target:
            k += 1
        i = max(k, i + 1)

    if not windows:
        windows.append({
            "id": "window_001",
            "start": 0.0,
            "end": round(float(video_duration), 3),
            "text": str(transcript_result.get("text", "") or ""),
        })
    return windows


def dedupe_overlapping(shorts, ratio=0.25):
    """Drop clips that overlap an already-kept clip by ``ratio`` of the
    shorter one, keeping the higher ``predicted_score`` (earlier on a tie).

    0.25, not 0.5: two clips sharing 25 s of 59 s (42%) both survived the old
    bar and the same punchline came out twice in one batch.

    The detail prompt's DIVERSITY rule is the only thing that stopped two
    clips from sharing the same seconds, and a rule is not a guarantee: two
    picks from one window can cover the same moment with different edges,
    and the user then downloads the same short twice. Survivors come back in
    transcript order, like ``trim_to_best``.
    """
    def _score(c):
        try:
            return float(c.get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    def _span(c):
        try:
            return float(c.get("start", 0)), float(c.get("end", 0))
        except (TypeError, ValueError, AttributeError):
            return 0.0, 0.0

    indexed = list(enumerate(shorts))
    kept = []
    for idx, clip in sorted(indexed, key=lambda p: (-_score(p[1]), p[0])):
        s, e = _span(clip)
        clash = False
        for _, other in kept:
            os_, oe = _span(other)
            overlap = min(e, oe) - max(s, os_)
            shorter = max(1e-6, min(e - s, oe - os_))
            if overlap > 0 and overlap / shorter >= ratio:
                clash = True
                break
        if not clash:
            kept.append((idx, clip))
    return [c for _, c in sorted(kept, key=lambda p: p[0])]


# --- Sentence-anchored clip boundaries -------------------------------------
# The detail pass used to receive a whole window (start, end, text) and return
# absolute seconds. The model cannot see where a sentence falls inside 180 s,
# so 8 of 14 boundaries on the 22-nov-2026 pass landed mid-phrase; word
# snapping only pulls ±1.5 s to the nearest word. Let the model name the first
# and last complete sentence, and derive the time span here.

LEAD_SILENCE = 0.15      # s kept before the first word
TAIL_SILENCE = 0.35      # s kept after the last word
MAX_JOIN_GAP = 2.0       # s; a longer pause ends the clip instead of bridging it


def sentence_list(segments):
    """Whisper segments as ``[{id, start, end, text}]`` with stable global ids."""
    out = []
    for seg in segments or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        out.append({"id": f"s{len(out) + 1:04d}", "start": start, "end": end, "text": text})
    return out


_ID_RE = re.compile(r"^(?:s(?:entence)?[\s_-]*)?(\d+)$", re.IGNORECASE)


def normalize_sentence_id(raw):
    """``"s201"``, ``"S0201"``, ``"201"`` and ``"sentence 201"`` -> ``"s0201"``.

    Small models copy ids loosely; a miss is not a wrong clip, just a fallback
    to the numeric proposal, but normalising keeps the common shapes working.
    """
    # The prompt shows ids in brackets, so models often echo "[s0041]",
    # "s0041." or '"s0041"'. Strip that wrapper before matching; a bare
    # bracketed id used to miss and silently fall back to the numeric times.
    text = str(raw or "").strip().strip("[](){}<>\"'.,;: \t")
    match = _ID_RE.match(text)
    return f"s{int(match.group(1)):04d}" if match else text


def resolve_sentence_span(start_id, end_id, sentences, video_duration,
                          min_duration=15.0, max_duration=60.0,
                          window_start=None, window_end=None):
    """``(start, end)`` covering whole sentences from ``start_id`` to ``end_id``.

    Grows or shrinks by COMPLETE sentences until the band is met, so a cut
    never lands inside a phrase. Padding never steals a neighbour: it is
    clamped to half of the real gap, so contiguous Whisper segments produce a
    clean cut instead of 150/350 ms of the next sentence.

    Returns ``None`` when the ids are unknown or outside the candidate window,
    when the span would bridge a pause longer than ``MAX_JOIN_GAP``, when one
    sentence alone is longer than ``max_duration``, or when the band cannot be
    met without leaving the window — the caller then falls back.
    """
    sentences = list(sentences or [])
    index = {s["id"]: k for k, s in enumerate(sentences)}
    i = index.get(normalize_sentence_id(start_id))
    j = index.get(normalize_sentence_id(end_id))
    if i is None or j is None:
        return None
    if j < i:
        i, j = j, i
    duration = max(0.0, float(video_duration or 0))
    lo = 0.0 if window_start is None else max(0.0, float(window_start))
    hi = duration if window_end is None else min(duration, float(window_end))
    if hi <= lo:
        return None
    # Only sentences that sit fully inside the candidate window: the detail
    # pass never saw the others, and a clip must not jump to another group.
    pool = {k for k, s in enumerate(sentences)
            if s["start"] >= lo - 1e-6 and s["end"] <= hi + 1e-6}
    if i not in pool or j not in pool:
        return None

    # A span that bridges a long internal pause is not one moment: bounce it
    # so the caller can fall back instead of shipping 30 s of dead air.
    for k in range(i, j):
        if sentences[k + 1]["start"] - sentences[k]["end"] > MAX_JOIN_GAP:
            return None

    def lead(k):
        gap = sentences[k]["start"] - sentences[k - 1]["end"] if k > 0 else LEAD_SILENCE
        return min(LEAD_SILENCE, max(0.0, gap) / 2)

    def tail(k):
        gap = sentences[k + 1]["start"] - sentences[k]["end"] if k + 1 < len(sentences) else TAIL_SILENCE
        return min(TAIL_SILENCE, max(0.0, gap) / 2)

    def span(a, b):
        return (max(0.0, sentences[a]["start"] - lead(a)),
                min(duration, sentences[b]["end"] + tail(b)))

    def joins(k):  # sentence k flows into k + 1 without a long pause
        return k + 1 in pool and sentences[k + 1]["start"] - sentences[k]["end"] <= MAX_JOIN_GAP

    # A Whisper segment is not always a sentence: it often stops at a comma.
    # When the window has punctuation at all, prefer edges on segments that
    # close a sentence. Without punctuation this is skipped and the span is
    # built from whole segments exactly as before.
    punctuated = any(closes_sentence(sentences[k]["text"]) for k in pool)

    start, end = span(i, j)
    while end - start > max_duration and j > i:
        j -= 1
        start, end = span(i, j)
    if end - start > max_duration:
        return None
    if punctuated and not closes_sentence(sentences[j]["text"]):
        for k in range(j - 1, i - 1, -1):
            if closes_sentence(sentences[k]["text"]):
                s2, e2 = span(i, k)
                if e2 - s2 >= min_duration:
                    j, start, end = k, s2, e2
                break

    while end - start < min_duration:
        if (j + 1 in pool
                and sentences[j + 1]["start"] - sentences[j]["end"] <= MAX_JOIN_GAP):
            nj = j + 1
            ns, ne = span(i, nj)
            if ne - ns <= max_duration:
                j, end = nj, ne
                continue
        if (i - 1 in pool
                and sentences[i]["start"] - sentences[i - 1]["end"] <= MAX_JOIN_GAP):
            ni = i - 1
            ns, ne = span(ni, j)
            if ne - ns <= max_duration:
                i, start = ni, ns
                continue
        break

    if punctuated:
        # End: run on to the segment that closes the sentence.
        k = j
        while not closes_sentence(sentences[k]["text"]) and joins(k):
            s2, e2 = span(i, k + 1)
            if e2 - s2 > max_duration:
                break
            k += 1
        if closes_sentence(sentences[k]["text"]):
            j = k
            start, end = span(i, j)
        # Start: step back to just after the previous sentence closed.
        k = i
        while k - 1 in pool and not closes_sentence(sentences[k - 1]["text"]) and joins(k - 1):
            s2, e2 = span(k - 1, j)
            if e2 - s2 > max_duration:
                break
            k -= 1
        if k == 0 or (k - 1 >= 0 and closes_sentence(sentences[k - 1]["text"])):
            i = k
            start, end = span(i, j)

    if end - start < min_duration or end - start > max_duration:
        return None
    return (round(start, 3), round(end, 3))


_CLOSE_RE = re.compile(r"[.!?\u2026][\"'\u00bb\u201d)\]]*\s*$")


def closes_sentence(text):
    """True when a transcript segment ends a sentence (. ! ? or ellipsis)."""
    return bool(_CLOSE_RE.search(str(text or "")))


def snap_clip_to_words(start, end, words, video_duration,
                       min_duration=15.0, max_duration=60.0,
                       search_window=1.5, max_lead=0.35, max_tail=0.45):
    """
    Snap Gemini-proposed clip boundaries onto real word boundaries plus a bit
    of the surrounding silence. LLMs are bad at millisecond arithmetic; the
    word-level timestamps are ground truth, so cuts land in pauses instead of
    mid-word.

    words: [{'w','s','e'}, ...] for the whole video, sorted by start.
    Returns (start, end); falls back to the input if no words are nearby or
    snapping cannot satisfy the duration bounds.
    """
    original = (round(float(start), 3), round(float(end), 3))
    if not words:
        return original

    starts = [float(w.get("s", 0)) for w in words]
    ends = [float(w.get("e", 0)) for w in words]

    # START: snap to the nearest word start, then lead into the silence before it.
    new_start = float(start)
    candidates = [s for s in starts if abs(s - new_start) <= search_window]
    if candidates:
        word_start = min(candidates, key=lambda s: abs(s - new_start))
        prev_ends = [e for e in ends if e <= word_start]
        if prev_ends:
            gap = max(0.0, word_start - max(prev_ends))
            lead = min(max_lead, gap / 2)
        else:
            lead = max_lead
        new_start = max(0.0, word_start - lead)

    # END: snap to the nearest word end, then trail into the silence after it.
    new_end = float(end)
    candidates = [e for e in ends if abs(e - new_end) <= search_window]
    if candidates:
        word_end = min(candidates, key=lambda e: abs(e - new_end))
        next_starts = [s for s in starts if s >= word_end]
        if next_starts:
            gap = max(0.0, min(next_starts) - word_end)
            tail = min(max_tail, gap / 2)
        else:
            tail = max_tail
        new_end = min(float(video_duration), word_end + tail)

    # Repair duration bounds while staying on word boundaries.
    if new_end - new_start < min_duration:
        target = new_start + min_duration
        later = sorted(e for e in ends if e >= target)
        if later and later[0] - new_start <= max_duration:
            new_end = min(float(video_duration), later[0] + 0.2)
        else:
            return original
    if new_end - new_start > max_duration:
        target = new_start + max_duration
        earlier = [e for e in ends if new_start < e <= target]
        new_end = (max(earlier) + 0.2) if earlier else target
        new_end = min(new_end, new_start + max_duration, float(video_duration))

    if new_end <= new_start or new_end - new_start < min_duration:
        return original
    return (round(new_start, 3), round(new_end, 3))


# --- Channel brief: keep a pass on the channel's niche ----------------------
# A strong hook from the wrong niche still scored high: on a 23-nov-2026 pass
# three of seven clips came from an essay-writing tutorial for an AI channel.
# The brief is opt-in per job (CHANNEL_BRIEF); without it nothing below runs.
# A per-clip "on_brief" flag from the model did not work (it signed true on
# every clip, intro included), so the floor is structural: windows under it
# never reach the detail pass.

ON_BRIEF_SCORE_FLOOR = 40
MAX_BRIEF_CHARS = 1500


def clean_brief(raw):
    """The brief as one bounded line of text, or "" when there is none."""
    text = " ".join(str(raw or "").split())
    return text[:MAX_BRIEF_CHARS]


def on_brief_windows(scored, floor=ON_BRIEF_SCORE_FLOOR):
    """Scored windows at or above the floor, best first. With a brief, these
    are the only ones the detail pass (and its min-clips retry) may use."""
    keep = []
    for w in scored or []:
        try:
            score = float(w.get("score", 0))
        except (TypeError, ValueError):
            continue
        if score >= floor:
            keep.append(w)
    return sorted(keep, key=lambda w: float(w.get("score", 0)), reverse=True)


# Openings that announce instead of deliver ("hoy les voy a enseñar...").
_ANNOUNCE_RE = re.compile(
    r"^\W*(?:(?:y|bueno|pues|entonces|hola)\W+)*"
    r"(?:hoy\s+(?:les|os|te)\s+voy\s+a|(?:les|os|te)\s+voy\s+a\s+(?:enseñar|mostrar|explicar|contar)"
    r"|vamos\s+a\s+(?:ver|aprender|hablar)|en\s+este\s+v[ií]deo"
    r"|today\s+i(?:'m|\s+am)?\s+(?:going\s+to|gonna)|in\s+this\s+video)",
    re.IGNORECASE)


def is_announcement(text):
    """True when a sentence only announces what comes later."""
    return bool(_ANNOUNCE_RE.search(str(text or "")))


def drop_leading_announcements(start, end, sentences, min_duration):
    """``(start, end)`` with announcement sentences cut from the opening.

    Returns the span unchanged when it does not open on an announcement,
    the trimmed span when what remains still meets ``min_duration``, and
    ``None`` when the clip is (almost) all announcement.
    """
    inside = [s for s in sentences or []
              if s["start"] >= start - 0.5 and s["end"] <= end + 0.5]
    k = 0
    while k < len(inside) and is_announcement(inside[k]["text"]):
        k += 1
    if k == 0:
        return (start, end)
    if k >= len(inside):
        return None
    new_start = max(start, inside[k]["start"] - 0.15)
    if end - new_start < float(min_duration):
        return None
    return (round(new_start, 3), end)


# --- ASR glossary: fix recurring mis-hearings before anything reads them ----
# Whisper small writes "chat GPT", "desintergrados"... and the text is burned
# as is. Applied once to the transcript, so selection and subtitles see the
# same fixed words. One-to-one swaps keep the word's timestamps; a multi-word
# match collapses into one word spanning first start to last end.

_EDGE_PUNCT = ".,;:!?¡¿\"'«»()[]…"


def parse_glossary(raw):
    """``{"wrong": "right"}`` from JSON or ``"wrong => right; a => b"``."""
    text = str(raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        import json
        try:
            data = json.loads(text)
        except ValueError:
            return {}
        return {str(k).strip(): str(v).strip() for k, v in data.items()
                if str(k).strip() and str(v).strip()}
    out = {}
    for part in re.split(r"[;\n]", text):
        if "=>" in part:
            wrong, right = part.split("=>", 1)
            if wrong.strip() and right.strip():
                out[wrong.strip()] = right.strip()
    return out


def _norm_token(word):
    return str(word or "").strip().strip(_EDGE_PUNCT).lower()


def _apply_to_words(words, entries):
    out, i = [], 0
    while i < len(words):
        for tokens, right in entries:
            k = len(tokens)
            if [_norm_token(w["word"]) for w in words[i:i + k]] == tokens:
                first, last = words[i], words[i + k - 1]
                raw_last = str(last["word"]).rstrip()
                trail = raw_last[len(raw_last.rstrip(_EDGE_PUNCT)):]
                lead = str(first["word"])[:len(str(first["word"])) - len(str(first["word"]).lstrip())]
                out.append({**first, "word": f"{lead}{right}{trail}", "end": last["end"]})
                i += k
                break
        else:
            out.append(words[i])
            i += 1
    return out


def apply_glossary(transcript, glossary):
    """A new transcript with every glossary entry fixed in words and text."""
    if not glossary or not transcript:
        return transcript
    entries = sorted(((_norm_token_list(w), r) for w, r in glossary.items()),
                     key=lambda e: len(e[0]), reverse=True)
    entries = [e for e in entries if e[0]]
    patterns = [(re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, w.split())) + r"(?!\w)",
                            re.IGNORECASE), r) for w, r in glossary.items() if w.split()]

    def fix_text(text):
        for rx, right in patterns:
            text = rx.sub(right, text)
        return text

    segments = []
    for seg in transcript.get("segments") or []:
        segments.append({**seg,
                         "text": fix_text(str(seg.get("text") or "")),
                         "words": _apply_to_words(list(seg.get("words") or []), entries)})
    return {**transcript, "segments": segments,
            "text": fix_text(str(transcript.get("text") or ""))}


def _norm_token_list(phrase):
    return [t for t in (_norm_token(p) for p in str(phrase).split()) if t]
