"""
tasks.py -- the five benchmark tasks, as prompt builders + strict parsers.

Every task reads ONLY from `dataset/dataset_anonymized.csv` and only from the
fields a model under test is allowed to see. `story_id`, `tau`, `lis`, both
order columns, both scene columns, the notes and `story_graph.story_time` are
corpus-side gold and never enter a prompt to a model under test.

  T1 transpose_cued   fabula + named category  -> told order      (RQ2, = C2)
  T2 transpose_free   fabula, no instruction   -> told order      (RQ1, = C1)
  T3 reconstruct      gold syuzhet             -> true chronology (recoverability)
  T4 classify         gold syuzhet             -> category label  (analysis)
  T5 prose            fabula + named category  -> a written story (generation)

The category descriptions are reproduced verbatim from the pilot notebook in
which T1 was first defined, so that T1 is the pilot's own task, run at scale
across models rather than one model at a time.
"""
import json
import re

CATEGORY_DESCRIPTIONS = {
    "ACHRONY": (
        "You cannot confidently assign chronological ranks because the text gives no dates, "
        "no ageing, no causal signposts, and no other ordering cues. Stop here only if the order "
        "is genuinely unrecoverable, not merely withheld until a late reveal."
    ),
    "CHRONICLE": (
        "The chronological ranks run 1, 2, 3 ... n in told order. Simultaneous events appearing "
        "in either order still qualify."
    ),
    "RETROGRADE": (
        "The ranks run n ... 3, 2, 1 throughout, and there is no present-day scene the narrative "
        "keeps returning to."
    ),
    "ZIGZAG": (
        "The scenes split into two time periods that alternate on a regular beat; each period "
        "progresses in its own consistent direction; and the paired scenes are semantically related "
        "(echo, explanation, or contrast). One of the two strands may run backward."
    ),
    "ANALEPSIS": (
        "One timeline dominates (it holds more than about three quarters of the scenes) and the "
        "remaining scenes are insertions from earlier than that timeline."
    ),
    "PROLEPSIS": (
        "Same as above, but the insertions come from later than the dominant timeline. The inserted "
        "scene must be a real scene the reader can later place, not a vague premonition."
    ),
    "SYLLEPSIS": (
        "There is no dominant timeline, but the scenes form blocks grouped by a nameable non-temporal "
        "principle: place, character, theme, or object. You must be able to state the principle."
    ),
}

ORDER_SYS = ("You are a narrative-ordering engine. Reorder the supplied numbered events so\n"
             "their told order realizes the requested nonlinear category. Use every event index "
             "exactly once.\nReturn only one line of space-separated integers. Do not return prose, "
             "JSON, labels, punctuation,\nanalysis, markdown, or an explanation.")

FREE_SYS = ("You are a storyteller planning how to tell a story. You choose the order in which the "
            "events will be narrated. Use every event index exactly once.\n"
            "Return only one line of space-separated integers. Do not return prose, JSON, labels, "
            "punctuation, analysis, markdown, or an explanation.")

SOLVE_SYS = ("You are a narrative analyst. You reconstruct the true chronological order of a story's "
             "events from a telling that may present them out of order. Use every event index exactly "
             "once.\nReturn only one line of space-separated integers. Do not return prose, JSON, "
             "labels, punctuation, analysis, markdown, or an explanation.")

CLASSIFY_SYS = ("You are a narratologist. You name the way a telling departs from chronology. "
                "Answer with exactly one label from the supplied list and nothing else.")

PROSE_SYS = ("You are an expert fiction writer. Use only the supplied events. "
             "Return story prose only: no title, analysis, category name, bullets, or markdown.")


def split_events(cell):
    """Split the corpus's canonical pipe-separated event representation."""
    events = [e.strip() for e in cell.split(" | ")]
    if not 10 <= len(events) <= 20 or any(not e for e in events):
        raise ValueError("expected 10-20 non-empty ' | '-separated events")
    if len(events) != len(set(events)):
        raise ValueError("events must be unique to recover a permutation exactly")
    return events


def _numbered(events):
    return "\n".join(f"{i}. {e}" for i, e in enumerate(events, 1))


# ------------------------------------------------------------------ prompts
def prompt_transpose_cued(row):
    """T1 -- the replicate notebook's own task: fabula + target category -> told order."""
    ev = split_events(row["fabula"])
    cat = row["category"]
    return f"""TARGET NONLINEAR CATEGORY
{cat}

CATEGORY DESCRIPTION
{CATEGORY_DESCRIPTIONS[cat]}

FABULA EVENTS IN CHRONOLOGICAL ORDER
{_numbered(ev)}

Choose the told order. Output exactly {len(ev)} space-separated indices: a permutation of
{' '.join(str(i) for i in range(1, len(ev) + 1))}
No index may be missing or repeated. Output the indices only."""


def prompt_transpose_free(row):
    """T2 -- RQ1. No structure named, no hint that reordering is wanted or unwanted."""
    ev = split_events(row["fabula"])
    return f"""STORY EVENTS IN CHRONOLOGICAL ORDER
{_numbered(ev)}

You are going to tell this story. Decide the order in which you would narrate these
events to a reader. Output exactly {len(ev)} space-separated indices: a permutation of
{' '.join(str(i) for i in range(1, len(ev) + 1))}
No index may be missing or repeated. Output the indices only."""


def prompt_reconstruct(row):
    """T3 -- recoverability. The model sees the telling and must rebuild the fabula."""
    told = split_events(row["syuzhet"])
    return f"""SCENES IN THE ORDER THE STORY TELLS THEM
{_numbered(told)}

These scenes may be told out of chronological order. Work out the order in which the
events actually happened in the story world, earliest first.
Output exactly {len(told)} space-separated indices: a permutation of
{' '.join(str(i) for i in range(1, len(told) + 1))}
Each index refers to a scene above. No index may be missing or repeated.
Output the indices only."""


def prompt_classify(row):
    """T4 -- can the model name the structure it is looking at?"""
    told = split_events(row["syuzhet"])
    defs = "\n".join(f"{c}: {CATEGORY_DESCRIPTIONS[c]}" for c in CATEGORY_DESCRIPTIONS)
    return f"""SCENES IN THE ORDER THE STORY TELLS THEM
{_numbered(told)}

CATEGORY DEFINITIONS
{defs}

Which single category best describes how this telling departs from chronology?
Answer with exactly one of: {', '.join(CATEGORY_DESCRIPTIONS)}
Output the label only."""


def prompt_prose(row):
    """T5 -- the generation arm: an actual written story, not a permutation."""
    ev = split_events(row["fabula"])
    cat = row["category"]
    return f"""TARGET NONLINEAR CATEGORY
{cat}

CATEGORY DESCRIPTION
{CATEGORY_DESCRIPTIONS[cat]}

STORY EVENTS IN CHRONOLOGICAL ORDER
{_numbered(ev)}

Write this as a short story of 550-750 words whose told order realizes the target
category. Cover every event above. Do not number the scenes and do not mention the
category. Output the story prose only."""


def prompt_prose_free(row):
    """T5b -- the prose counterpart of RQ1: no structure named."""
    ev = split_events(row["fabula"])
    return f"""STORY EVENTS IN CHRONOLOGICAL ORDER
{_numbered(ev)}

Write this as a short story of 550-750 words. Cover every event above.
Do not number the scenes. Output the story prose only."""


# ------------------------------------------------- the graph arm (G0/G1/G2)
# Three prompts that differ in exactly one thing: what the model is told about
# the story beyond its events. Same task, same word target, same number of API
# calls, so a difference in the output is a difference made by the scaffold.
#
# G2 exists because G1 is not a clean intervention on its own. A REVEALS edge is
# only legal when the explanation is told *after* the event it explains, and a
# question is only legal when its answer arrives later in the telling, so the
# graph silently pins 5.5% of all told-order pairs. Without G2 -- which hands
# over the told order and nothing else -- a G1 gain cannot be separated from
# "the model was given part of the answer".

def _graph_for_prompt(row):
    """The graph as the writer may see it: no story_id, no tau, no told order,
    and no story_time, which would hand over the chronological ranking the
    numbering already conveys."""
    g = row["story_graph"]
    if not g:
        raise ValueError(f"{row['story_id']} has no story_graph")
    ev = []
    for n in g["events"]:
        ev.append({k: v for k, v in n.items()
                   if k not in ("story_time", "told_at") and v not in ({}, [], "")})
    return {"characters": g["characters"], "events": ev, "edges": g["edges"]}


GRAPH_GLOSS = {
    "base":      "`core`/`content` describe each event",
    "grounding": "`participants`/`location` ground it",
    "state":     "`pre`/`post` are world-state facts that must stay consistent",
    "epistemic": ("`learns` records what each character and the reader comes to "
                  "know"),
    "questions": ("`questions`/`raises`/`resolves` mark what the telling withholds "
                  "and where it pays off"),
    "edges":     ("the typed edges give causes, plants, withheld explanations and "
                  "shared material"),
}
ABLATED_CLAUSE = {"A1_no_causal": None, "A2_no_state": "state",
                  "A3_no_epistemic": "epistemic", "A4_no_questions": "questions",
                  "A5_no_grounding": "grounding"}


def _gloss(drop=None):
    """The field gloss, describing exactly the layers the payload still carries.

    An ablation that removed a layer but kept describing it would be telling the
    model about something that is not there; keeping the clause set in step with
    the payload is what makes the removal the only difference.
    """
    keys = [k for k in GRAPH_GLOSS if k != drop]
    parts = [GRAPH_GLOSS[k] for k in keys]
    return ("The graph below annotates the same events. "
            + ", ".join(parts[:-1]) + ", and " + parts[-1] + ".")


def prompt_prose_graph(row):
    """G1 -- fabula, target category, and the full typed narrative graph."""
    ev = split_events(row["fabula"])
    cat = row["category"]
    return f"""TARGET NONLINEAR CATEGORY
{cat}

CATEGORY DESCRIPTION
{CATEGORY_DESCRIPTIONS[cat]}

STORY EVENTS IN CHRONOLOGICAL ORDER
{_numbered(ev)}

NARRATIVE GRAPH
The graph below annotates the same events. `core`/`content` describe each event,
`participants`/`location` ground it, `pre`/`post` are world-state facts that must
stay consistent, `learns` records what each character and the reader comes to
know, `questions`/`raises`/`resolves` mark what the telling withholds and where it
pays off, and the typed edges give causes, plants, withheld explanations and
shared material.
{json.dumps(_graph_for_prompt(row), ensure_ascii=False)}

Write this as a short story of 550-750 words whose told order realizes the target
category. Cover every event above. Use the graph to keep the world state and the
characters' knowledge consistent, and to decide what to withhold and when to pay
it off. Do not number the scenes, do not mention the category, and do not quote
the graph. Output the story prose only."""


# Leave-one-layer-out ablations. Each removes exactly one layer from the same
# graph G1 receives, so a drop against G1 is attributable to that layer rather
# than to a shorter prompt or a different task.
LAYER_ABLATIONS = {
    "A1_no_causal":    "drop CAUSES and FORESHADOWS edges",
    "A2_no_state":     "drop pre/post world-state flags",
    "A3_no_epistemic": "drop the learns layer",
    "A4_no_questions": "drop questions/raises/resolves and REVEALS edges",
    "A5_no_grounding": "drop participants and location",
}


def _ablate(graph, which):
    """Return the graph with one layer removed. Removal is real -- the key is
    gone, not blanked and not paraphrased back in elsewhere."""
    g = json.loads(json.dumps(graph))
    drop_node, drop_edge = set(), set()
    if which == "A1_no_causal":
        drop_edge = {"CAUSES", "FORESHADOWS"}
    elif which == "A2_no_state":
        drop_node = {"pre", "post"}
    elif which == "A3_no_epistemic":
        drop_node = {"learns"}
    elif which == "A4_no_questions":
        drop_node = {"questions", "raises", "resolves"}
        drop_edge = {"REVEALS"}
    elif which == "A5_no_grounding":
        drop_node = {"participants", "location"}
    else:
        raise ValueError(which)
    g["events"] = [{k: v for k, v in n.items() if k not in drop_node}
                   for n in g["events"]]
    g["edges"] = [e for e in g["edges"] if e["type"] not in drop_edge]
    return g


def _prompt_prose_ablated(row, which):
    ev = split_events(row["fabula"])
    cat = row["category"]
    graph = _ablate(_graph_for_prompt(row), which)
    gloss = _gloss(ABLATED_CLAUSE[which])
    if which == "A1_no_causal":
        gloss = gloss.replace("the typed edges give causes, plants, withheld "
                              "explanations and shared material",
                              "the typed edges give withheld explanations and "
                              "shared material")
    return f"""TARGET NONLINEAR CATEGORY
{cat}

CATEGORY DESCRIPTION
{CATEGORY_DESCRIPTIONS[cat]}

STORY EVENTS IN CHRONOLOGICAL ORDER
{_numbered(ev)}

NARRATIVE GRAPH
{gloss}
{json.dumps(graph, ensure_ascii=False)}

Write this as a short story of 550-750 words whose told order realizes the target
category. Cover every event above. Use the graph to keep the world state and the
characters' knowledge consistent, and to decide what to withhold and when to pay
it off. Do not number the scenes, do not mention the category, and do not quote
the graph. Output the story prose only."""


def prompt_prose_order(row):
    """G2 -- fabula, target category, and the gold told order. No semantics."""
    ev = split_events(row["fabula"])
    cat = row["category"]
    told = " ".join(str(i) for i in row["gold_order"])
    return f"""TARGET NONLINEAR CATEGORY
{cat}

CATEGORY DESCRIPTION
{CATEGORY_DESCRIPTIONS[cat]}

STORY EVENTS IN CHRONOLOGICAL ORDER
{_numbered(ev)}

TOLD ORDER
Narrate the events in exactly this order: {told}

Write this as a short story of 550-750 words that narrates the events in the told
order given above. Cover every event. Do not number the scenes and do not mention
the category. Output the story prose only."""


def prompt_align(events, story):
    """Judge-side: align a generated story back to the numbered fabula events,
    in NARRATED order, so the prose arm yields a permutation like T1/T2 do."""
    return f"""STORY EVENTS (chronological, numbered)
{_numbered(events)}

STORY
{story}

The story narrates these events in some order, possibly not the chronological one.
List the event numbers in the order the STORY NARRATES them. Include each number
exactly once; if the story handles an event only in passing, still place it where
it is first narrated.
Output exactly {len(events)} space-separated indices and nothing else."""


TASKS = {
    "T1_transpose_cued":  (ORDER_SYS,    prompt_transpose_cued),
    "T2_transpose_free":  (FREE_SYS,     prompt_transpose_free),
    "T3_reconstruct":     (SOLVE_SYS,    prompt_reconstruct),
    "T4_classify":        (CLASSIFY_SYS, prompt_classify),
    "T5_prose_cued":      (PROSE_SYS,    prompt_prose),
    "T5b_prose_free":     (PROSE_SYS,    prompt_prose_free),
    # the graph arm: G0 is prompt_prose under a name that says what it controls for
    "G0_fabula_only":     (PROSE_SYS,    prompt_prose),
    "G1_full_graph":      (PROSE_SYS,    prompt_prose_graph),
    "G2_order_only":      (PROSE_SYS,    prompt_prose_order),
    **{k: (PROSE_SYS, (lambda w: (lambda row: _prompt_prose_ablated(row, w)))(k))
       for k in LAYER_ABLATIONS},
}

PERMUTATION_TASKS = ["T1_transpose_cued", "T2_transpose_free", "T3_reconstruct"]


# ------------------------------------------------------------------ parsers
def parse_permutation(raw, n, lenient=True):
    """Strict permutation parse. `lenient` tolerates the wrappers reasoning
    models add (fences, a 'Answer:' line, a trailing sentence) but never
    tolerates a wrong multiset -- a non-permutation is a failed row, not a
    repaired one."""
    text = str(raw).strip()
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    text = text.replace("```json", " ").replace("```", " ")

    strict = re.sub(r"[\d\s,\[\]()\-]+", "", text)
    if not strict:
        nums = [int(x) for x in re.findall(r"\d+", text)]
        if len(nums) == n and sorted(nums) == list(range(1, n + 1)):
            return nums, "strict"
    if not lenient:
        raise ValueError(f"expected indices only, got: {text[:160]!r}")

    # Longest run of integers anywhere in the reply that forms a permutation.
    for line in [text] + text.splitlines()[::-1]:
        nums = [int(x) for x in re.findall(r"\d+", line)]
        for start in range(0, max(1, len(nums) - n + 1)):
            window = nums[start:start + n]
            if len(window) == n and sorted(window) == list(range(1, n + 1)):
                return window, "lenient"
    raise ValueError(f"no permutation of 1..{n} in reply: {text[:200]!r}")


def parse_label(raw, allowed):
    text = re.sub(r"<think>.*?</think>", " ", str(raw), flags=re.S | re.I).upper()
    hits = [c for c in allowed if re.search(rf"\b{c}\b", text)]
    if len(hits) == 1:
        return hits[0]
    if hits:                      # several named: take the last mention
        return max(hits, key=lambda c: text.rfind(c))
    raise ValueError(f"no category label in reply: {str(raw)[:160]!r}")
