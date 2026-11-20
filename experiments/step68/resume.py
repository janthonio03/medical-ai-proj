"""STEP68: bounded Neutral-only resume; original STEP60C source stays unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

HOME_RUN = Path("results/step60c-full-256-slurmfix")
PLAN_ID = "4fb7dbc47e6cb6a84fff5efafa8684c1e9bf3e372db40508a7a42d7835b35608"
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path("CAST-step60c-eval").resolve()))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(ok, message):
    if not ok:
        raise ValueError(message)

def common():
    from repro import step60c_common as c
    return c

def original_ticket():
    c = common()
    p = HOME_RUN / "batches/4e67ae3a4db94791a365ade68d21ade7.json"
    return c.read(p)

def load_plan(ticket):
    c = common()
    p = c.validate_plan(ticket["repo"], ticket["plan_dir"])
    check(p["plan_sha256"] == ticket["plan_sha256"] == PLAN_ID,
          "Original plan identity changed")
    return p

def selected(plan, keys):
    tasks = [t for t in plan["tasks"] if t["task_key"] in set(keys)]
    check(len(tasks) == len(keys) == len(set(keys)), "Task selection mismatch")
    check(all(t["condition"] == "paper-neutral" for t in tasks),
          "Non-Neutral task rejected")
    return tasks

def pin_check(ticket):
    pin = ticket["step68_neutral_only"]
    check(sha(__file__) == pin["adapter_sha256"], "Adapter source changed")
    check(sha(HERE / "job.sh") == pin["launcher_sha256"], "Launcher changed")
    p = Path(pin["request_file"])
    check(sha(p) == pin["request_sha256"], "Resume request changed")
    request = common().read(p)
    check(request["task_keys"] == pin["task_keys"], "Ticket selection differs")
    return request

def worker(path):
    c = common()
    ticket = c.read(path)
    pin_check(ticket)
    plan = load_plan(ticket)
    allowed = ticket["step68_neutral_only"]["task_keys"]
    selected(plan, allowed)
    check(ticket["scope"] == "full" and ticket["workers"] == 4,
          "Unexpected worker configuration")
    from repro import step60c_worker as w
    original_scope = w.scoped_tasks
    def scope(p, scope_name):
        check(p["plan_sha256"] == PLAN_ID and scope_name == "full",
              "Worker plan/scope changed")
        expected = {t["task_key"] for t in original_scope(p, scope_name)}
        check(set(allowed) <= expected, "Selected keys absent from full plan")
        return selected(p, allowed)
    w.scoped_tasks = scope
    print("STEP68_SELECTED_NEUTRAL_TASKS=" + str(len(allowed)), flush=True)
    return w.run_ticket(Path(path).resolve())

def submit():
    check(Path.cwd() == Path("/nas2/data/janthonio03/maip/cast-rebuild"), "Run from cast-rebuild")
    c = common()
    from repro import step60c_submit as s, step60c_results as r
    old = original_ticket()
    plan = load_plan(old)
    run = Path(old["run_dir"])
    rows = s.queue_snapshot()
    check(not rows, "Existing queued/running user jobs: inspect before requesting four GPUs")
    s.no_step60b(plan, rows)
    s.prior_jobs_inactive(run, rows)
    want, good, missing, invalid, extra = r.scan(plan, run, "full")
    check(not invalid and not extra, "Invalid/extra completions: investigate")
    neutral = [t for t in want if t["condition"] == "paper-neutral"]
    keys = [t["task_key"] for t in neutral if t["task_key"] in set(missing)]
    check(len(neutral) == 1061 and len(keys) == 101,
          "Expected 1061 Neutral tasks with exactly 101 missing")
    check(sum(t["condition"] == "paper-spatial-temporal" for t in good) == 1061,
          "Spatial-Temporal cohort is incomplete")
    # No existing result attempts may be silently reinferred.
    for t in selected(plan, keys):
        folder = run / "samples" / t["sample_id"] / t["condition"] / "attempts"
        check(not any(folder.glob("*/result.json")),
              "Missing receipt has a result attempt: investigate " + t["task_key"])
    before = {p.name: sha(p) for p in (run / "completed").glob("*.json")}
    request = {
        "plan_sha256": PLAN_ID, "task_keys": keys,
        "completed_receipt_sha256_before": before,
        "created_unix": time.time(),
        "adapter_sha256": sha(__file__), "launcher_sha256": sha(HERE / "job.sh"),
    }
    # Unique request path preserves every earlier submission attempt.
    request_path = HERE / ("request-" + str(time.time_ns()) + ".json")
    c.write(request_path, request)
    original_scope = r.scoped_tasks
    def scope(p, name):
        tasks = original_scope(p, name)
        return [t for t in tasks if t["condition"] == "paper-neutral"] if name == "full" else tasks
    r.scoped_tasks = scope
    original_array = s.array_command
    def array(*args, **kwargs):
        argv = original_array(*args, **kwargs)
        return [("--wrap=exec " + shlex.join(["bash", str(HERE / "job.sh")]))
                if x.startswith("--wrap=") else x for x in argv]
    s.array_command = array
    original_write = s.write
    def write(path, value):
        if Path(path).parent == run / "batches" and value.get("status") == "submitting":
            value["step68_neutral_only"] = {
                "task_keys": keys, "adapter_sha256": sha(__file__),
                "launcher_sha256": sha(HERE / "job.sh"),
                "request_file": str(request_path), "request_sha256": sha(request_path),
            }
        return original_write(path, value)
    s.write = write
    result = s.submit(
        old["repo"], old["plan_dir"], old["run_dir"], "full", 4,
        smoke_run=old["smoke_run"], smoke_merge=old["smoke_merge"], resume=True,
    )
    print(json.dumps(result, indent=2), flush=True)
    if result.get("submitted"):
        c.write(HERE / "latest.json", {"ticket": result["ticket"]})
    return 0

def report(verify=False):
    c = common()
    pointer = c.read(HERE / "latest.json")
    ticket = c.read(pointer["ticket"])
    req = pin_check(ticket)
    plan = load_plan(ticket)
    run = Path(ticket["run_dir"])
    tasks = selected(plan, req["task_keys"])
    done = [t for t in tasks if c.valid_result(plan, run, t)]
    failures = list((run / "failed_in_batch" / ticket["batch"]).glob("*.json"))
    records = [c.read(p) for p in (run / "workers" / ticket["batch"]).glob("*.json")]
    starts = [x["started_unix"] for x in records]
    elapsed = time.time() - min(starts) if starts else 0
    rate = len(done) / elapsed if elapsed > 0 else 0
    recent = sum(c.completion(run, t).stat().st_mtime >= time.time()-3600 for t in done)
    print("JOB_ID:", ticket["array_job_id"])
    print("NEW_NEUTRAL_COMPLETE:", len(done), "/ 101")
    print("REMAINING:", 101-len(done), "FAILED_IN_BATCH:", len(failures))
    print("RECENT_1H_COMPLETE:", recent)
    print("ETA_HOURS:", round((101-len(done))/rate/3600, 2) if rate else "unknown")
    print("WORKERS:", [(x["worker_index"], x["status"], x["completed_here"], x["failed_here"]) for x in records])
    print("SLURM_LOGS:", run / "slurm")
    print("RESULTS:", run / "samples")
    if verify:
        from repro.step60c_results import scan
        check(len(done) == 101 and not failures, "Resume incomplete or task failures exist")
        check(len(records) == 4 and all(x["status"] == "finished" and x["failed_here"] == 0 for x in records),
              "Workers not all finished successfully")
        before = req["completed_receipt_sha256_before"]
        check(all((run/"completed"/name).is_file() and sha(run/"completed"/name) == h
                  for name,h in before.items()), "An old completion receipt changed")
        expected_new = {t["task_key"] + ".json" for t in tasks}
        after = {p.name for p in (run/"completed").glob("*.json")}
        check(after == set(before) | expected_new, "Unexpected completed tasks")
        want, good, missing, invalid, extra = scan(plan, run, "full")
        check(not invalid and not extra, "Invalid/extra completions after resume")
        counts = {name: sum(t["condition"] == name for t in good)
                  for name in plan["conditions"]}
        check(counts["paper-neutral"] == counts["paper-spatial-temporal"] == 1061,
              "Paired paper cohort incomplete")
        print("VALID_COMPLETIONS:", counts)
        print("STEP68_NEUTRAL_RESUME_VERIFIED")
        print("Historical FP32/Eager cohort only; full four-condition benchmark remains incomplete.")
    return 0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["submit", "worker", "status", "verify"])
    p.add_argument("ticket", nargs="?")
    args = p.parse_args()
    try:
        if args.mode == "submit": code = submit()
        elif args.mode == "worker":
            check(bool(args.ticket), "Worker ticket required")
            code = worker(args.ticket)
        else: code = report(args.mode == "verify")
    except Exception as e:
        print("STEP68_ERROR=" + type(e).__name__ + ": " + str(e), flush=True)
        raise
    raise SystemExit(code)
