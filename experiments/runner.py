"""
runner.py -- execute benchmark tasks against OpenRouter, resumably.

    python runner.py --models MODELS.txt --tasks T1,T2,T3,T4 [--limit N]

One JSONL file per (model, task) under experiments/raw/. Every completed row is
appended immediately, so an interrupted run resumes exactly where it stopped and
nothing is ever paid for twice. Failures are recorded as rows too -- a model that
cannot emit a permutation is a result, not a gap.
"""
import argparse, json, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tasks as T

HERE = Path(__file__).parent
RAW = HERE / "raw"
RAW.mkdir(exist_ok=True)
DATASET = HERE.parent / "dataset" / "dataset_anonymized.csv"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_RETRIES = 5
TIMEOUT = 300

# Uniform decoding settings, fixed before the run and applied to every model, so
# differences in the tables are differences between models rather than between
# configurations. `reasoning.effort=low` is a budget decision as much as a design
# one; 05_reasoning_effort measures what raising it changes.
DECODE = {
    "T1_transpose_cued": dict(max_tokens=8000, temperature=0.0, reasoning="low"),
    "T2_transpose_free": dict(max_tokens=8000, temperature=0.0, reasoning="low"),
    "T3_reconstruct":    dict(max_tokens=8000, temperature=0.0, reasoning="low"),
    "T4_classify":       dict(max_tokens=8000, temperature=0.0, reasoning="low"),
    "T5_prose_cued":     dict(max_tokens=4000, temperature=0.7, reasoning="low"),
    "T5b_prose_free":    dict(max_tokens=4000, temperature=0.7, reasoning="low"),
    "G0_fabula_only":    dict(max_tokens=4000, temperature=0.7, reasoning="low"),
    "G1_full_graph":     dict(max_tokens=4000, temperature=0.7, reasoning="low"),
    "G2_order_only":     dict(max_tokens=4000, temperature=0.7, reasoning="low"),
    **{k: dict(max_tokens=4000, temperature=0.7, reasoning="low")
       for k in ("A1_no_causal", "A2_no_state", "A3_no_epistemic",
                 "A4_no_questions", "A5_no_grounding")},
}
_print_lock = threading.Lock()
_file_locks = {}


# --------------------------------------------------------------- corpus IO
def load_corpus(limit=None):
    """Load only the columns a model under test is allowed to see, plus the
    gold columns the *scorer* needs (never the prompt builder)."""
    import csv
    csv.field_size_limit(10_000_000)
    rows = []
    with open(DATASET, newline="", encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append({
                "row": i,
                "story_id": r["story_id"],                 # gold: scoring key only
                "genre": r["genre"], "source": r["source"],
                "fabula": r["fabula"].strip(),
                "syuzhet": r["syuzhet"].strip(),
                "category": r["nonlinear_category"].strip().upper(),
                "gold_order": [int(x) for x in r["syuzhet_order"].split()],
                "gold_scene_order": [int(x) for x in r["fabula_scene_order"].split()],
                "tau": float(r["tau"]), "lis": float(r["lis"]),
                # Gold for every task except the G-arm, where it is the
                # intervention itself. prompt builders decide, not the loader.
                "story_graph": json.loads(r["story_graph"]) if r.get("story_graph") else None,
            })
    return rows[:limit] if limit else rows


# ------------------------------------------------------------------ client
class APIError(RuntimeError):
    def __init__(self, msg, status=None, retryable=False):
        super().__init__(msg)
        self.status, self.retryable = status, retryable


def call(model_id, system, user, api_key, max_tokens, temperature=None,
         require_zdr=False, extra=None):
    body = {"model": model_id, "stream": False, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if temperature is not None:
        body["temperature"] = temperature
    if require_zdr:
        body["provider"] = {"zdr": True}
    body.update(extra or {})

    req = Request(API_URL, data=json.dumps(body).encode(), method="POST",
                  headers={"Authorization": f"Bearer {api_key}",
                           "Content-Type": "application/json",
                           "X-Title": "Nonlinear Narrative Benchmark"})
    try:
        with urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode())
    except HTTPError as e:
        body_txt = e.read().decode("utf-8", "replace")
        raise APIError(f"HTTP {e.code}: {body_txt[:400]}", e.code,
                       e.code in {408, 409, 429, 500, 502, 503, 504}) from e
    except URLError as e:
        raise APIError(f"unreachable: {e.reason}", retryable=True) from e
    except TimeoutError as e:
        raise APIError("socket timeout", retryable=True) from e


def content_of(resp):
    c = resp["choices"][0]["message"].get("content")
    if isinstance(c, list):
        c = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    c = (c or "").strip()
    if not c:                                   # reasoning-only reply, no answer
        rc = resp["choices"][0]["message"].get("reasoning") or ""
        if rc:
            return str(rc).strip()
        raise ValueError(f"empty content (finish={resp['choices'][0].get('finish_reason')})")
    return c


# ------------------------------------------------------------------- worker
def run_one(spec, row, api_key):
    """One task on one story. Returns a result dict; never raises for model
    failure -- only for account-level errors that make continuing pointless."""
    task = spec["task"]
    dec = dict(DECODE[task], **spec.get("decode_override", {}))
    system, builder = T.TASKS[task]
    user = builder(row)
    n = len(T.split_events(row["fabula"] if task != "T3_reconstruct" else row["syuzhet"]))
    out = {"story_id": row["story_id"], "row": row["row"], "task": task,
           "model": spec["name"], "model_id": spec["id"], "category": row["category"],
           "replicate": spec.get("replicate", 1)}

    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            resp = call(spec["id"], system, user, api_key,
                        max_tokens=dec["max_tokens"], temperature=dec["temperature"],
                        require_zdr=spec.get("zdr", False),
                        extra={"reasoning": {"effort": dec["reasoning"]}})
            raw = content_of(resp)
            usage = resp.get("usage") or {}
            out.update({"latency_s": round(time.time() - t0, 2), "decode": dec,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "cost": usage.get("cost"), "attempts": attempt,
                        "raw": raw[:8000]})
            if task == "T4_classify":
                out["label"] = T.parse_label(raw, list(T.CATEGORY_DESCRIPTIONS))
            elif task.startswith(("T5", "G0", "G1", "G2", "A1", "A2", "A3", "A4", "A5")):
                out["text"] = raw
                out["word_count"] = len(raw.split())
            else:
                order, mode = T.parse_permutation(raw, n)
                out["order"], out["parse_mode"] = order, mode
            out["status"] = "ok"
            return out
        except APIError as e:
            last = e
            if e.status in (401, 402):                       # account-level: stop
                raise
            if not e.retryable and attempt >= 2:
                break
        except Exception as e:
            last = e                                          # parse / empty reply
        if attempt < MAX_RETRIES:
            time.sleep(min(30, 2.0 * 2 ** (attempt - 1)) + random.uniform(0, 0.75))

    out.update({"status": "failed", "error": f"{type(last).__name__}: {str(last)[:300]}",
                "attempts": MAX_RETRIES})
    return out


def out_path(model_name, task, rep=1):
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_name)
    tag = "" if rep == 1 else f"__r{rep}"
    return RAW / f"{safe}__{task}{tag}.jsonl"


def done_keys(path):
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                keys.add(r["story_id"])
    return keys


def append(path, rec):
    lock = _file_locks.setdefault(str(path), threading.Lock())
    with lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_model_task(spec, task, rows, api_key, workers, rep=1):
    path = out_path(spec["name"], task, rep)
    done = done_keys(path)
    todo = [r for r in rows if r["story_id"] not in done]
    if not todo:
        print(f"  {spec['name']:<26} {task:<18} complete ({len(done)} rows)")
        return {"ok": len(done), "failed": 0, "cost": 0.0}

    stats = {"ok": 0, "failed": 0, "cost": 0.0}
    sub = dict(spec, task=task, replicate=rep)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, sub, r, api_key): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except APIError as e:
                print(f"\n!! account-level error, aborting: {e}")
                raise
            append(path, rec)
            stats["ok" if rec["status"] == "ok" else "failed"] += 1
            stats["cost"] += rec.get("cost") or 0.0
            if i % 20 == 0 or i == len(todo):
                with _print_lock:
                    print(f"  {spec['name']:<26} {task:<18} r{rep} "
                          f"{i}/{len(todo)}  ok={stats['ok']} fail={stats['failed']} "
                          f"${stats['cost']:.4f}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--tasks", default="T1_transpose_cued,T2_transpose_free,"
                                       "T3_reconstruct,T4_classify")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-category", type=int, default=None,
                    help="stratified subsample: first N stories of each category")
    ap.add_argument("--only", default=None, help="comma-separated model names")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--replicates", type=int, default=1,
                    help="sample each (model, task, story) this many times")
    a = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set")

    specs = json.load(open(a.models, encoding="utf-8"))
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        specs = [s for s in specs if s["name"] in want]
    rows = load_corpus(a.limit)
    if a.per_category:
        seen, keep = {}, []
        for r in rows:
            c = r["category"]
            if seen.get(c, 0) < a.per_category:
                seen[c] = seen.get(c, 0) + 1
                keep.append(r)
        rows = keep
    task_list = [t.strip() for t in a.tasks.split(",") if t.strip()]
    print(f"{len(specs)} model(s) x {len(task_list)} task(s) x {len(rows)} stor(ies)\n")

    total = 0.0
    for spec in specs:
        for task in task_list:
            if task.startswith("T5") and not spec.get("prose", True):
                continue
            try:
                for rep in range(1, a.replicates + 1):
                    s = run_model_task(spec, task, rows, api_key, a.workers, rep)
                    total += s["cost"]
            except APIError:
                raise
            except Exception as e:
                print(f"  !! {spec['name']} {task} aborted: {type(e).__name__}: {str(e)[:200]}")
        print(f"  -- running total ${total:.4f}\n", flush=True)
    print(f"done. spend this run: ${total:.4f}")


if __name__ == "__main__":
    main()
