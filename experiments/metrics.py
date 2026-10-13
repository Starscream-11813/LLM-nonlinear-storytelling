"""
metrics.py -- deterministic scoring for the permutation-level benchmark.

Everything here is pure Python over a told order `t` (a list of 1-based
chronological ranks, in the order the telling presents them). No API, no judge,
no tunable weights -- which is what makes these numbers reproducible.

kendall_tau / lis_ratio restate the definitions used by
`dataset/build_dataset.py`.
They are re-implemented rather than imported so that the benchmark is an
independent restatement, not a restatement of the corpus build's assumptions
(the same reasoning `dataset/verify_dataset.py` gives for its own copies).
"""
from bisect import bisect_left
from itertools import combinations

CATEGORIES = ["ACHRONY", "CHRONICLE", "RETROGRADE", "ZIGZAG",
              "ANALEPSIS", "PROLEPSIS", "SYLLEPSIS"]

# Classes decidable from the permutation alone. SYLLEPSIS and ACHRONY turn on
# what the scenes are *about* (see README.md), so a structural
# classifier can only ever guess at them; they are reported separately.
STRUCTURAL_CATEGORIES = ["CHRONICLE", "RETROGRADE", "ZIGZAG",
                         "ANALEPSIS", "PROLEPSIS"]


# --------------------------------------------------------------- primitives
def kendall_tau(t):
    """+1 strictly chronological, -1 strictly reversed, ~0 scrambled."""
    n = len(t)
    if n < 2:
        return 1.0
    inv = sum(1 for i, j in combinations(range(n), 2) if t[i] > t[j])
    return 1.0 - 2.0 * inv / (n * (n - 1) / 2)


def lis_ratio(t):
    """Longest increasing subsequence / n. 1.0 = the telling never doubles back."""
    tails = []
    for x in t:
        k = bisect_left(tails, x)
        tails[k:k + 1] = [x]
    return len(tails) / len(t) if t else 0.0


def spearman(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def displacement(t):
    """Mean |told position - chronological position|, normalised by n.
    0 = chronological; ~0.33 = uniformly random for large n."""
    n = len(t)
    if n < 2:
        return 0.0
    return sum(abs((i + 1) - rank) for i, rank in enumerate(t)) / (n * n)


def is_identity(t):
    return t == list(range(1, len(t) + 1))


def is_reverse(t):
    return t == list(range(len(t), 0, -1))


# ------------------------------------------------------------------ scenes
def scene_sequence(told, event_scenes):
    """Map a told order of events to its told order of scene ids, runs collapsed.

    `event_scenes[i]` is the scene id of chronological event i+1.
    Returns (collapsed scene sequence, contiguity) where contiguity is the
    fraction of gold scenes whose events survive as one unbroken run -- the
    corpus constraint a telling has to respect for the scene reading to mean
    anything.
    """
    seq = [event_scenes[e - 1] for e in told]
    collapsed = [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]
    n_scenes = len(set(seq))
    contiguity = n_scenes / len(collapsed) if collapsed else 0.0
    return collapsed, contiguity


def scene_ranks(collapsed):
    """Renumber a collapsed scene sequence to ranks 1..M in chronological order,
    so it can be fed to kendall_tau exactly like an event order."""
    order = sorted(set(collapsed))
    pos = {s: i + 1 for i, s in enumerate(order)}
    return [pos[s] for s in collapsed]


# -------------------------------------------------------------- classifier
def _dominant_run(t, min_share=0.55):
    """Longest increasing subsequence, returned as a set of told positions.
    Stands in for the 'narrative present' -- the band the telling is based in."""
    n = len(t)
    tails, tails_idx, prev = [], [], [-1] * n
    for i, x in enumerate(t):
        k = bisect_left(tails, x)
        if k == len(tails):
            tails.append(x); tails_idx.append(i)
        else:
            tails[k], tails_idx[k] = x, i
        prev[i] = tails_idx[k - 1] if k else -1
    if not tails_idx:
        return set()
    chain, i = [], tails_idx[-1]
    while i != -1:
        chain.append(i); i = prev[i]
    chain.reverse()
    return set(chain) if len(chain) >= min_share * n else set()


def _alternation_score(t):
    """How regularly the telling alternates between an early band and a late
    band -- the ZIGZAG signature. Split at the rank median, then measure the
    fraction of adjacent told pairs that cross the split."""
    n = len(t)
    if n < 4:
        return 0.0
    mid = (n + 1) / 2
    band = [0 if x <= mid else 1 for x in t]
    switches = sum(1 for i in range(1, n) if band[i] != band[i - 1])
    balance = min(band.count(0), band.count(1)) / (n / 2)
    return (switches / (n - 1)) * balance


def _monotone_bands(t):
    """Are both bands internally monotone (either direction)? ZIGZAG requires it."""
    mid = (len(t) + 1) / 2
    ok = 0
    for want_low in (True, False):
        b = [x for x in t if (x <= mid) == want_low]
        if len(b) < 2:
            ok += 1
            continue
        up = sum(1 for i in range(1, len(b)) if b[i] > b[i - 1]) / (len(b) - 1)
        if up >= 0.8 or up <= 0.2:
            ok += 1
    return ok == 2


def classify(t):
    """Structural 7-way guess at the nonlinear category of a told order.

    Honest about its own limits: SYLLEPSIS and ACHRONY are content decisions,
    so they are only ever reached as a fallback. Calibrated against the 83 gold
    labels by `score.py`, and its accuracy there is reported as this metric's
    noise floor rather than assumed away.
    """
    n = len(t)
    if n < 3:
        return "CHRONICLE"
    tau = kendall_tau(t)

    if tau >= 0.80:
        return "CHRONICLE"
    if tau <= -0.60:
        return "RETROGRADE"

    alt, dom = _alternation_score(t), _dominant_run(t)
    if alt >= 0.62 and _monotone_bands(t):
        return "ZIGZAG"

    if dom:
        # Insertions: are the off-timeline scenes earlier or later than the
        # dominant band position they interrupt?
        chain_vals = [t[i] for i in sorted(dom)]
        earlier = later = 0
        for i, x in enumerate(t):
            if i in dom:
                continue
            before = [v for j, v in enumerate(chain_vals) if sorted(dom)[j] < i]
            ref = before[-1] if before else chain_vals[0]
            earlier += x < ref
            later += x > ref
        if earlier or later:
            return "ANALEPSIS" if earlier >= later else "PROLEPSIS"

    # Blocks: a few long forward runs jumping around => grouped by something
    # other than time. Structural shadow of SYLLEPSIS; not a proof of it.
    runs = 1 + sum(1 for i in range(1, n) if t[i] < t[i - 1])
    if runs <= max(2, n // 4) and lis_ratio(t) >= 0.45:
        return "SYLLEPSIS"
    return "ACHRONY"


# ------------------------------------------------------------------- bands
def build_bands(gold_rows):
    """Per-category (tau, LIS) bands read off the human-annotated corpus.

    This is the study's 'human reference band': a generated telling is scored
    by whether it lands inside the range real nonlinear works of that category
    actually occupy, rather than against a number anyone chose.
    """
    bands = {}
    for c in CATEGORIES:
        taus = [r["tau"] for r in gold_rows if r["category"] == c]
        lis = [r["lis"] for r in gold_rows if r["category"] == c]
        if taus:
            bands[c] = {"tau_min": min(taus), "tau_max": max(taus),
                        "lis_min": min(lis), "lis_max": max(lis), "n": len(taus)}
    return bands


def in_band(bands, category, tau, lis):
    b = bands.get(category)
    if not b:
        return None
    return bool(b["tau_min"] <= tau <= b["tau_max"] and
                b["lis_min"] <= lis <= b["lis_max"])
