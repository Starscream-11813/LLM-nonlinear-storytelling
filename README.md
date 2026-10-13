# LLM-nonlinear-storytelling

Data and code for a study of how large language models handle **nonlinear storytelling**:
telling the events of a story in an order different from the order in which they happen.
This repository accompanies a paper under anonymous review.

It contains three things:

1. **A corpus of 83 nonlinear narratives**, each annotated with the same set of events in
   two orders, plus a hand-authored narrative graph for every work.
2. **A benchmark run across 19 LLMs**, with every model and judge reply stored, and the code
   that scores them.
3. **The human-evaluation responses.**

No API key is needed to inspect or reproduce anything reported. Every table in
`experiments/results/` is regenerated offline from the stored replies in `experiments/raw/`.

---

## Contents

```
dataset/
  movies/                  83 source annotations, one JSON per work (with titles)
  movies_anon/             83 anonymised annotations with gold story graphs (S01-S83)
  dataset.csv              the titled corpus, built from movies/
  dataset_anonymized.csv   the anonymised corpus, built from movies_anon/ (what models saw)
  anonymization_key.csv    story_id -> original title and year
  build_dataset.py         validates movies/ and builds dataset.csv
  build_anonymized.py      validates movies_anon/ against its source; builds the anonymised CSV
  verify_dataset.py        independent round-trip check of both CSVs
experiments/
  tasks.py                 prompts and strict output parsers
  metrics.py               Kendall tau, LIS ratio and the other structural measures
  runner.py                resumable API runner for the benchmark tasks
  judge.py, judge_grapharm.py         LLM judges: prose-to-event alignment, PDS, pairwise
  rcg.py, rcg_baseline.py             retrospective-coherence experiment and its no-text baseline
  dedate.py, recover.py               de-dating and the recoverability experiment
  score*.py, analysis.py              offline scoring: raw/ -> results/
  models.json, models_effort.json     the evaluated models
  dedated_events.json      de-dated event lists used by the recoverability experiment
  raw/                     144 JSONL files: every model and judge reply
  results/                 49 derived tables
humaneval/
  responses2.csv           human-evaluation responses
```

## Quick start: reproduce everything offline

Python 3.10 or newer. Only `pandas` is needed beyond the standard library, and only for scoring.

```bash
python -m pip install -r requirements.txt

# 1. Validate the corpus (writes nothing)
cd dataset
python build_dataset.py --check
python build_anonymized.py --check
python verify_dataset.py

# 2. Rebuild every table in experiments/results/ from the stored replies
cd ../experiments
python score.py            # T1-T4 tables, per-row scores, classifier calibration
python analysis.py         # breakdowns by category, cost and effort; common-subset tables
python score_prose.py      # prose arm (T5/T5b) and inter-judge agreement
python score_grapharm.py   # graph-scaffolding arm (G0, G1, G2, A1-A5)
python score_rcg.py        # retrospective coherence gain and reveal placement
python score_recover.py    # recoverability experiment and the no-story floor
```

Step 2 makes no network calls. We verified that it regenerates all 49 files in
`experiments/results/` with identical contents. On Windows, pandas writes CRLF line
endings, so `git status` may list the regenerated tables as modified even though
`git diff` shows no change.

---

## 1. The corpus

### What is annotated

Narratology distinguishes the **fabula**, the order in which events happen in the story
world, from the **syuzhet**, the order in which the work tells them. Every work in the
corpus is annotated with one set of events, written once in chronological order, plus the
order in which the work presents them. Because both orders describe the same event set,
the pair is a **permutation by construction**.

| | |
|---|---:|
| works | 83 (71 films, 7 novels, 2 plays, 2 musicals, 1 short story) |
| events | 1,433 (12-20 per work) |
| scenes | 1,221 |
| characters | 857 |
| typed graph edges | 3,478 |
| reader-question threads | 657 |
| character and reader knowledge claims | 2,504 |

### Files

The JSON files are the source of truth; both CSVs are build artifacts.

- `movies/NN-slug.json` holds `title`, `release_year`, `genre`, `source`,
  `events_chronological`, `syuzhet_order`, `event_scenes`, `syuzhet_notes` and
  `fabula_notes`. `NN` is the work's position in the canonical list in `build_dataset.py`.
- `movies_anon/NN.json` holds the anonymised twin of the same work: `story_id`,
  `source_index`, `genre`, `source`, `events_chronological`, `syuzhet_order`,
  `event_scenes`, `nonlinear_category`, `nonlinear_category_reason`, the notes, and
  `story_graph`.

`dataset.csv` columns: `title, release_year, genre, source, fabula, syuzhet, fabula_order,
syuzhet_order, fabula_scene_order, syuzhet_scene_order, tau, lis, syuzhet_notes,
fabula_notes`.

`dataset_anonymized.csv` columns: `story_id, genre, source, fabula, syuzhet, fabula_order,
syuzhet_order, fabula_scene_order, syuzhet_scene_order, tau, lis, nonlinear_category,
syuzhet_notes, fabula_notes, story_graph`.

| column | content |
|---|---|
| `fabula`, `syuzhet` | the event bullets in chronological and in told order, joined by `" \| "` |
| `fabula_order`, `syuzhet_order` | the same two orders as space-separated 1-based event indices. Events are numbered by chronological position, so `fabula_order` is always `1 2 ... n`, and the k-th entry of `syuzhet_order` is the chronological index of the k-th event told |
| `fabula_scene_order`, `syuzhet_scene_order` | one scene id per bullet, aligned with the two orders above. Collapsing runs of repeats gives the scene-level permutation |
| `tau` | Kendall's tau of the told order against chronology: +1 chronological, -1 fully reversed |
| `lis` | length of the longest increasing subsequence of the told order, divided by n: 1.0 when the telling never doubles back |
| `nonlinear_category` | one of the seven structure labels below |
| `story_graph` | the gold narrative graph, as compact JSON |

Reading a row:

```python
import csv
row = next(csv.DictReader(open("dataset/dataset_anonymized.csv", encoding="utf-8", newline="")))
events = row["fabula"].split(" | ")                      # chronological order
told = [int(i) for i in row["syuzhet_order"].split()]    # a permutation of 1..n
telling = [events[i - 1] for i in told]                  # == row["syuzhet"].split(" | ")
```

**Scenes.** A scene is a stretch the work presents as one continuous unit. Several events
can share a scene. Every scene is required to be contiguous in *both* orders, which is what
makes the scene sequence a permutation as well.

### Nonlinear categories

Each work carries one of seven labels describing how its told scene order departs from
chronology, with a one-sentence justification in `nonlinear_category_reason`. Every
category has at least ten works (Analepsis 22, Syllepsis 11, and 10 each for the rest).

| label | told order |
|---|---|
| `CHRONICLE` | runs forward; scenes presented as simultaneous may arrive in either order |
| `RETROGRADE` | runs backward throughout, with no present the telling returns to |
| `ZIGZAG` | two bands alternating on a regular beat, each internally monotone |
| `ANALEPSIS` | one timeline (the narrative present) dominates; the rest are insertions from earlier |
| `PROLEPSIS` | as above, but the insertions are genuine flash-forwards from later |
| `SYLLEPSIS` | no dominant timeline; blocks grouped by a non-temporal principle (place, character, theme, object) |
| `ACHRONY` | none of the above; the told order follows no statable rule |

All `ACHRONY` works are achronic by shuffle: their chronology is recoverable. Works whose
chronology cannot be reconstructed at all are excluded from the corpus.

### The story graph

The bullets say *what* happens. The graph says *why*, *who knew what when*, and *what the
telling withholds*. It has three parts:

- `characters`: name -> `{"archetype": ...}`, using Vogler's eight roles (Hero, Mentor,
  Ally, Herald, Trickster, Shapeshifter, Threshold Guardian, Shadow). Where a finer label
  was annotated it is kept in `archetype_detail`.
- `events`: node `ei` is bullet `i`, with `story_time == i`.

  | field | content |
  |---|---|
  | `core` | subject-verb-object triple |
  | `content` | one to three sentences elaborating the bullet |
  | `participants`, `location` | who is present and where; participants are declared in `characters` |
  | `pre`, `post` | world-state flags before and after the event |
  | `learns` | per character, and for `reader`, the propositions the event teaches |
  | `raises`, `resolves` | question ids; question text is in `questions` |

- `edges`: four typed relations. `σ` is chronological position and `told` is told position.

  | type | constraint enforced by the build |
  |---|---|
  | `CAUSES` | σ(from) < σ(to) |
  | `FORESHADOWS` | σ(from) < σ(to) |
  | `REVEALS` | σ(from) < σ(to) **and** told(from) > told(to): the explanation is told after what it explains, backed by a matching raises/resolves pair |
  | `SHARES` | undirected; the endpoints share a participant, or the edge names what they share in `on` |

The build also replays `post` flags in chronological order and requires every `pre` to be
satisfied, and requires every question to be raised exactly once and resolved exactly
once. The told position is deliberately not stored on nodes: it is derivable from
`syuzhet_order`, and storing it would put the answer to the reordering tasks inside the
graph.

### Anonymisation

The anonymised twin is what models under test were given. Relative to the source it keeps
the event count, `syuzhet_order`, scene structure, `tau`, `lis`, `genre`, `source`, causal
structure and granularity identical, so the two CSVs join row-for-row. It replaces every
character name, place, organisation, in-world title, quotation, distinctive prop,
signature number, real person and real historical event, and drops `title` and
`release_year` in favour of `story_id`. `build_anonymized.py` refuses to emit a story that
contains a proper noun from its source, shares a bullet verbatim with its source, or
disagrees with its source on any structural column.

`dataset.csv`, `anonymization_key.csv`, and the `story_time` field of the graph are the
answer key. **None of them should appear in a prompt to a model under test.**

Anonymisation removes surface recall cues but cannot hide a premise. A work whose concept
is unique remains recognisable from its structure, so contamination is reduced rather than
eliminated.

### Validation

- `build_dataset.py --check` validates every source annotation: `syuzhet_order` is a
  permutation of `1..n`, `n` is between 10 and 20, scenes are numbered chronologically and
  contiguous in both orders, and the remaining fields are well-formed.
- `build_anonymized.py --check` validates every anonymised annotation and its graph against
  the source, including the leak checks above.
- `verify_dataset.py` reads both CSVs the way a consumer would. It rebuilds each
  permutation from the prose columns alone, recomputes `tau` and `lis` with its own
  implementation, re-derives the graph anchors, checks the scene columns, and requires the
  two files to agree on every column they share. It exits non-zero on any problem.

---

## 2. The benchmark

### Models

19 models accessed through OpenRouter: 9 closed and 10 open-weight. `models.json` lists
each model's ID, lab and access type. `models_effort.json` lists four of them re-run at
high reasoning effort.

### Tasks

| id | input | output |
|---|---|---|
| `T1_transpose_cued` | events in chronological order + a named category | a told order |
| `T2_transpose_free` | events in chronological order | a told order |
| `T3_reconstruct` | events in the gold told order | the chronology |
| `T4_classify` | events in the gold told order | one of the seven categories |
| `T5_prose_cued` / `T5b_prose_free` | events (+ category) | a written story |

A model under test never sees `story_id`, either order column, either scene column, `tau`,
`lis`, the notes, the story graph, or the title key. T3 and T4 are given the gold told
order because reading it is the task.

Decoding is fixed across models: `temperature=0`, `reasoning.effort=low`,
`max_tokens=8000` for the ordering tasks; `temperature=0.7`, `max_tokens=4000` for prose.

### Further experiments

**Graph scaffolding.** A 550-750 word story realising a named category, under eight
conditions that differ only in what accompanies the event list:

| condition | given beyond the events and category |
|---|---|
| `G0_fabula_only` | nothing |
| `G1_full_graph` | the full story graph |
| `G2_order_only` | the gold told order (an oracle control, since the graph partly constrains the told order) |
| `A1_no_causal` | the graph without `CAUSES`/`FORESHADOWS` |
| `A2_no_state` | the graph without `pre`/`post` |
| `A3_no_epistemic` | the graph without `learns` |
| `A4_no_questions` | the graph without questions and `REVEALS` |
| `A5_no_grounding` | the graph without `participants`/`location` |

The ablation prompts are identical to `G1` except for the removed layer and its line of
explanation. Prose experiments use 14 stories, two per category.

**Retrospective coherence gain (RCG).** For each generated story, a solver reconstructs
the chronology from the text before the story's designated reveal and from the full text,
over the same events. RCG is the difference in Kendall tau. A paired guess task asks
whether the withheld cause could be named before it is revealed. `rcg_baseline.py` runs
the same solver with no story at all.

**Recoverability.** A 2×2 on reconstruction: the solver either sees the events shuffled or
in told order, with the events either as written or de-dated (explicit time anchors
removed by a guarded rewrite, stored in `dedated_events.json`). The four cells are
`R0_orig_notext`, `R1_orig_told`, `R2_dedated_notext` and `R3_dedated_told`.

Prose order is recovered by an LLM judge that aligns each story back to its events. A
second judge aligns a subsample independently, and `results/interjudge.json` reports their
agreement.

### Scoring

All scores are computed from the permutation alone.

- **tau**, **LIS**: how far the told order departs from chronology.
- **band hit**: whether a telling's (tau, LIS) lies inside the range occupied by real works
  of the requested category. The bands come from the corpus, so no threshold is hand-set.
- **agreement with gold**: tau between the model's telling and the work's own telling.
- **scene contiguity**: the fraction of gold scenes kept as one unbroken run.
- **exact match** (T3): whether the reconstructed chronology is entirely correct.

Reference rows are scored through the same code: `[baseline] identity`,
`[baseline] reverse`, `[baseline] random` and `[reference] human (gold)`. When pooling over
models under test, exclude rows whose model name starts with `[` or ends with `[high]`.
`[reference] human (gold)` is the annotation of the published work, not newly collected
human writing.

### Stored replies: `experiments/raw/`

One JSONL file per run, one JSON object per line.

| file | contents |
|---|---|
| `<model>__<task>.jsonl` | benchmark replies. Model names are made filename-safe, so the high-effort re-run `Claude Opus 5 [high]` becomes `Claude_Opus_5__high_`. A `__r2` suffix marks a second replicate |
| `judge_<judge>__<instrument>.jsonl` | prose-arm judging |
| `gjudge_<judge>__<instrument>.jsonl` | graph-arm judging; instruments are `align`, `pds` and `pairwise` |
| `rcg_<solver>.jsonl`, `rcg_none_<solver>.jsonl` | RCG cells and the no-text baseline |
| `recover.jsonl` | the recoverability 2×2 |

Generation records include `story_id`, `task`, `model`, `model_id`, `category`, `decode`,
token counts, `cost`, the unedited reply in `raw`, the parsed result (`order` for ordering
tasks, `text` and `word_count` for prose), `parse_mode` and `status`. `parse_mode` is
`strict` when the reply contains only the indices, and `lenient` when they had to be located
inside wrapper text such as a code fence, a reasoning block or an answer line. A reply that
does not contain an exact permutation is recorded as a failed row; it is never repaired.
`skipped` RCG cells are stories whose gold graph designates no reveal, or whose reveal is
narrated with fewer than three events before it.

### Derived tables: `experiments/results/`

| files | written by |
|---|---|
| `table_T1`-`table_T4`, `per_row.csv`, `calibration.json`, `classifier_calibration.csv` | `score.py` |
| `breakdown_*.csv`, `common_subset_*.csv` | `analysis.py` |
| `table_T5_*.csv`, `prose_*_rows.csv`, `interjudge.json` | `score_prose.py` |
| `grapharm_*.csv` | `score_grapharm.py` |
| `rcg_*.csv` | `score_rcg.py` |
| `recover_*.csv` | `score_recover.py` |

`common_subset_*.csv` repeat each comparison on the stories every model answered, because
refusals are not evenly distributed across models.

---

## 3. Human evaluation: `humaneval/responses2.csv`

Long format: one row per recorded value. 71 participants with pseudonymous IDs `1`-`71`,
13,823 rows. The stories are the 14 used in the prose experiments, written by 11 of the
generator models. Participants never saw which condition produced a story.

| column | content |
|---|---|
| `participant_id` | pseudonymous participant number |
| `submitted_at` | UTC timestamp |
| `block` | `background`, `literacy`, `pairwise` or `single` |
| `item_index` | position of the item within its block |
| `task_id` | item identifier; for literacy items, the corpus `story_id` |
| `kind` | comparison kind (pairwise) or condition (single) |
| `story_id`, `model`, `category` | the corpus story, the generating model and its nonlinear category |
| `condition_left`, `condition_right` | condition shown as Story A and Story B (pairwise); the condition read (single) |
| `words_left`, `words_right` | word counts of the two stories (pairwise); of the story (single) |
| `measure`, `value` | what was recorded, and the answer |
| `rt_ms` | time on the item in milliseconds, where recorded |

### `background`

`reads_fiction` and `writing_experience` (multiple choice), and `user_agent_width`, the
browser viewport width in pixels.

### `literacy`: reading the story-curve diagram

Each item shows the curve of a corpus story's gold telling, which plots told position
against chronological position, and asks three questions. `value` is the 0-based index of
the option chosen. Every answer is a deterministic function of the story's
`syuzhet_order` `o`, with `n` events:

| measure | question | options | correct |
|---|---|---|---|
| `literacy_choice_0` | Does the telling open at, or near, the earliest event? | `0` yes, `1` no | `1` if `o[0] > 0.6n`, else `0` |
| `literacy_choice_1` | How many times does the telling jump backwards? | `sorted({max(0,b-2), b, b+2, max(0,b-1)+4})`, where `b` is the number of positions `j` with `o[j] < o[j-1]` | the index of `b` |
| `literacy_choice_2` | Is the last scene shown also the last event? | `0` yes, `1` no | `0` if `o[-1] == n`, else `1` |

### `pairwise`: blind comparison of two tellings

Both stories come from the same model telling the same events under two conditions.

| `kind` | conditions compared |
|---|---|
| `cued_vs_free` | `T5_prose_cued` vs `T5b_prose_free` |
| `graph_vs_none` | `G1_full_graph` vs `G0_fabula_only` |
| `graph_vs_order` | `G1_full_graph` vs `G2_order_only` |
| `graph_vs_no_causal` | `G1_full_graph` vs `A1_no_causal` |
| `graph_vs_no_state` | `G1_full_graph` vs `A2_no_state` |
| `graph_vs_no_epistemic` | `G1_full_graph` vs `A3_no_epistemic` |
| `graph_vs_no_questions` | `G1_full_graph` vs `A4_no_questions` |
| `graph_vs_no_grounding` | `G1_full_graph` vs `A5_no_grounding` |

| measure | value |
|---|---|
| `better` | which is the better story overall: `A`, `B` or `tie` |
| `confusing` | which was harder to follow: `A`, `B` or `tie` |
| `ending` | whose ending tied things together better: `A`, `B` or `tie` |
| `confidence` | how sure the participant is of the `better` choice: 1 (a coin toss) to 5 (completely sure) |
| `reason` | optional free text |

`A` is the left story and `B` the right, so the chosen condition is `condition_left` for
`A` and `condition_right` for `B`. Measures ending in `_repeat` record a hidden repeat of an
earlier pair shown with the sides swapped, for measuring rater consistency.

### `single`: close reading of one story

`kind` and `condition_left` give the condition: `T5_prose_cued`, `G1_full_graph` or
`G0_fabula_only`.

| measure | value |
|---|---|
| `reading_time` | time spent reading the story, in `rt_ms` |
| `pds_authenticity`, `pds_empathy`, `pds_engagement`, `pds_emotion_provocation`, `pds_narrative_complexity` | Psychological Depth Scale items, 1 to 5 |
| `chronology_order` | the participant's ordering of the story's shuffled events, as space-separated chronological indices, earliest first. The correct answer is `1 2 ... n` |
| `question_status_0`-`question_status_3` | for each of up to four questions from the story's gold graph, taken in event order: `answered`, `open` or `not_raised` |
| `check_anchored`, `check_spoiler`, `check_jump_unjustified`, `check_knowledge`, `check_world`, `check_uncaused` | `1` if the participant reported the problem: a scene that could not be placed, a premature reveal, an unmotivated time jump, impossible character knowledge, a physical inconsistency, an unexplained event |
| `authorship_guess` | `human`, `machine` or `unsure`. Every story shown was machine-written |

---

## 4. Re-running generation (optional)

Inspecting or reproducing the reported results does not require this. The stored replies
are the record of the experiments, and re-running will not reproduce them exactly: prose is
sampled at temperature 0.7, and hosted models change over time.

These scripts call the API and incur cost: `runner.py`, `judge.py`, `judge_grapharm.py`,
`rcg.py`, `rcg_baseline.py`, `recover.py` and `dedate.py`. They read the key from the
environment:

```bash
export OPENROUTER_API_KEY=...
cd experiments
python runner.py --tasks T1_transpose_cued --only "Claude Opus 5"
```

`runner.py` is resumable: it skips every cell already present in `raw/`, so to generate
fresh replies, move or empty `raw/` first. Useful options are `--tasks`, `--only`,
`--per-category` and `--replicates`.

## License

MIT. See [LICENSE](LICENSE).
