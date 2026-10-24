#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
from openai import OpenAI

# Update here if there are changes in models/pricing.
PRICING = {
    "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
}

LOG_LOCK = Lock()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def next_run_dir(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in root.iterdir():
        if p.is_dir() and re.fullmatch(r"run_\d+", p.name):
            nums.append(int(p.name.split("_")[1]))
    n = max(nums, default=0) + 1
    run_dir = root / f"run_{n:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sample_size_95_10(population):
    """95% confidence, ±10% margin, p=.5, finite population correction."""
    if population <= 0:
        return 0
    z = 1.96
    p = 0.5
    e = 0.10
    n0 = (z * z * p * (1 - p)) / (e * e)
    n = n0 / (1 + ((n0 - 1) / population))
    return min(population, math.ceil(n))


def load_taxonomy(path: Path):
    df = pd.read_csv(path)
    required = {"task", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.dropna(subset=["task"]).copy()
    df["task"] = df["task"].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    if df["task"].duplicated().any():
        dups = df.loc[df["task"].duplicated(), "task"].tolist()
        raise ValueError(f"Duplicate taxonomy Task names in {path}: {dups}")
    return df


def load_original_tasks(path: Path):
    df = pd.read_csv(path)
    if "task" not in df.columns:
        raise ValueError(f"{path} must contain a 'task' column")
    df = df.dropna(subset=["task"]).copy()
    df["task"] = df["task"].astype(str).str.strip()
    df = df[df["task"] != ""].copy()
    # One row per unique original Task. If duplicates somehow exist, sum frequency.
    if "frequency" in df.columns:
        df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0)
        agg = {c: "first" for c in df.columns if c not in {"task", "frequency"}}
        agg["frequency"] = "sum"
        df = df.groupby("task", as_index=False).agg(agg)
    else:
        df = df.drop_duplicates(subset=["task"], keep="first")
        df["frequency"] = pd.NA
    return df.reset_index(drop=True)


def discover_domains(taxonomy_run_dir: Path, domain_dir: Path):
    found = []
    for folder in sorted(p for p in taxonomy_run_dir.iterdir() if p.is_dir()):
        taxonomy_path = folder / "taxonomy.csv"
        if not taxonomy_path.exists():
            continue
        original_path = domain_dir / f"{folder.name}_tasks.csv"
        if not original_path.exists():
            raise FileNotFoundError(
                f"Found taxonomy folder {folder.name}, but original file does not exist: {original_path}"
            )
        found.append({
            "folder": folder.name,
            "taxonomy_path": taxonomy_path,
            "original_path": original_path,
        })
    if not found:
        raise FileNotFoundError(f"No domain folders containing taxonomy.csv found in {taxonomy_run_dir}")
    return found


def build_prompt(prompt_beginning, domain_name, taxonomy_df, tasks):
    taxonomy_lines = []
    for _, row in taxonomy_df.iterrows():
        taxonomy_lines.append(f'- {row["task"]}: {row["description"]}')
    original_lines = [f"- {task}" for task in tasks]
    return (
        prompt_beginning.strip()
        + f"\n\nDOMAIN:\n{domain_name}\n\n"
        + "PREDEFINED TASKS:\n"
        + "\n".join(taxonomy_lines)
        + "\n\nORIGINAL TASKS TO ASSIGN:\n"
        + "\n".join(original_lines)
    )


def assignment_schema(allowed_labels):
    # Structured Outputs prevents new assigned labels at the schema level.
    return {
        "type": "json_schema",
        "name": "task_assignments",
        "strict": True,
        "schema": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_task": {"type": "string"},
                                "assigned_task": {
                                    "type": "string",
                                    "enum": list(allowed_labels) + ["Other"],
                                },
                            },
                            "required": ["original_task", "assigned_task"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["assignments"],
                "additionalProperties": False,
            },
    }


def extract_response_text(response):
    """Extract text from a Responses API response."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            value = getattr(content, "text", None)
            if value:
                chunks.append(value)

    return "\n".join(chunks).strip()


def extract_usage(response):
    """Extract token usage from a Responses API response."""
    usage = getattr(response, "usage", None)

    if usage is None:
        return 0, 0, 0

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    cached_input_tokens = 0
    input_details = getattr(usage, "input_tokens_details", None)

    if input_details is not None:
        cached_input_tokens = int(
            getattr(input_details, "cached_tokens", 0) or 0
        )

    return input_tokens, cached_input_tokens, output_tokens


def calculate_cost(model, input_tokens, cached_input_tokens, output_tokens):
    rates = PRICING.get(model)

    if not rates:
        return None

    uncached = max(0, input_tokens - cached_input_tokens)

    cost = (
        uncached * rates["input"]
        + cached_input_tokens * rates["cached_input"]
        + output_tokens * rates["output"]
    ) / 1_000_000

    return cost


def append_csv(path: Path, fieldnames, row):
    with LOG_LOCK:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def log_validation(path, domain, attempt, batch_id, original_task, flag, returned_value="", detail=""):
    append_csv(
        path,
        ["timestamp", "domain", "attempt", "batch_id", "original_task", "flag", "returned_value", "detail"],
        {
            "timestamp": now_iso(),
            "domain": domain,
            "attempt": attempt,
            "batch_id": batch_id,
            "original_task": original_task,
            "flag": flag,
            "returned_value": returned_value,
            "detail": detail,
        },
    )


def call_model(client, model, full_prompt, schema):
    """
    Call a GPT-5-family model using the Responses API.

    Temperature and reasoning effort are intentionally omitted
    so the model uses the same default inference configuration
    as the earlier Step 1 GPT-5 model comparison.
    """
    start = time.time()

    response = client.responses.create(
        model=model,
        input=full_prompt,
        text={
            "format": schema
        },
    )

    elapsed = time.time() - start

    content = extract_response_text(response)

    if not content:
        raise ValueError("The model returned no text output.")

    parsed = json.loads(content)

    (
        input_tokens,
        cached_input_tokens,
        output_tokens
    ) = extract_usage(response)

    cost = calculate_cost(
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens
    )

    usage = {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    return parsed, content, usage, cost, elapsed


def validate_response(input_tasks, parsed, allowed, domain, attempt, batch_id, validation_log):
    expected = list(input_tasks)
    expected_set = set(expected)
    rows = parsed.get("assignments", []) if isinstance(parsed, dict) else []

    by_original = {}
    for row in rows:
        if not isinstance(row, dict):
            log_validation(validation_log, domain, attempt, batch_id, "", "malformed_row", detail=str(row))
            continue
        original = str(row.get("original_task", "")).strip()
        assigned = str(row.get("assigned_task", "")).strip()
        if original not in expected_set:
            log_validation(validation_log, domain, attempt, batch_id, original, "unexpected_original_task", assigned)
            continue
        by_original.setdefault(original, []).append(assigned)

    valid = {}
    failed = []
    for task in expected:
        values = by_original.get(task, [])
        if len(values) == 0:
            log_validation(validation_log, domain, attempt, batch_id, task, "missing_assignment")
            failed.append(task)
            continue
        if len(values) > 1:
            log_validation(validation_log, domain, attempt, batch_id, task, "duplicate_assignment", " | ".join(values))
            failed.append(task)
            continue
        assigned = values[0]
        if assigned not in allowed and assigned != "Other":
            log_validation(validation_log, domain, attempt, batch_id, task, "invalid_label", assigned)
            failed.append(task)
            continue
        valid[task] = assigned
        log_validation(validation_log, domain, attempt, batch_id, task, "valid_other" if assigned == "Other" else "valid")
    return valid, failed


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_batch(client, model, prompt_text, domain_name, taxonomy_df, tasks, attempt, batch_id,
              domain_out, api_log, validation_log, error_log):
    allowed = taxonomy_df["task"].tolist()
    full_prompt = build_prompt(prompt_text, domain_name, taxonomy_df, tasks)
    schema = assignment_schema(allowed)
    prompt_dir = domain_out / "prompts"
    raw_dir = domain_out / "raw_responses"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"attempt_{attempt}_batch_{batch_id:04d}"
    (prompt_dir / f"{stem}.txt").write_text(full_prompt, encoding="utf-8")

    try:
        parsed, raw, usage, cost, elapsed = call_model(client, model, full_prompt, schema)
        (raw_dir / f"{stem}.json").write_text(raw, encoding="utf-8")
        valid, failed = validate_response(
            tasks, parsed, set(allowed), domain_name, attempt, batch_id, validation_log
        )
        append_csv(
            api_log,
            ["timestamp", "domain", "attempt", "batch_id", "n_input_tasks", "status",
             "prompt_tokens", "cached_tokens", "completion_tokens", "total_tokens",
             "estimated_cost_usd", "elapsed_seconds", "n_valid", "n_failed"],
            {
                "timestamp": now_iso(), "domain": domain_name, "attempt": attempt,
                "batch_id": batch_id, "n_input_tasks": len(tasks), "status": "success",
                "prompt_tokens": usage["input_tokens"],
                "cached_tokens": usage["cached_input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["total_tokens"],
                "estimated_cost_usd": "" if cost is None else f"{cost:.8f}",
                "elapsed_seconds": f"{elapsed:.3f}", "n_valid": len(valid), "n_failed": len(failed),
            },
        )
        return {"valid": valid, "failed": failed, "cost": cost or 0.0}
    except Exception as e:
        with LOG_LOCK:
            with error_log.open("a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] domain={domain_name} attempt={attempt} batch={batch_id} error={repr(e)}\n")
        for task in tasks:
            log_validation(validation_log, domain_name, attempt, batch_id, task, "api_or_parse_error", detail=repr(e))
        append_csv(
            api_log,
            ["timestamp", "domain", "attempt", "batch_id", "n_input_tasks", "status",
             "prompt_tokens", "cached_tokens", "completion_tokens", "total_tokens",
             "estimated_cost_usd", "elapsed_seconds", "n_valid", "n_failed"],
            {
                "timestamp": now_iso(), "domain": domain_name, "attempt": attempt,
                "batch_id": batch_id, "n_input_tasks": len(tasks), "status": "error",
                "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "estimated_cost_usd": "", "elapsed_seconds": "", "n_valid": 0, "n_failed": len(tasks),
            },
        )
        return {"valid": {}, "failed": list(tasks), "cost": 0.0}


def run_batches_concurrently(client, model, prompt_text, domain_name, taxonomy_df, tasks, attempt,
                             batch_size, workers, domain_out, api_log, validation_log, error_log):
    batches = list(chunks(tasks, batch_size))
    all_valid = {}
    all_failed = []
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                run_batch, client, model, prompt_text, domain_name, taxonomy_df, batch,
                attempt, idx + 1, domain_out, api_log, validation_log, error_log
            ): (idx + 1, batch)
            for idx, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_id, batch = futures[fut]
            result = fut.result()
            all_valid.update(result["valid"])
            all_failed.extend(result["failed"])
            total_cost += result["cost"]
            print(f"  attempt {attempt} batch {batch_id}/{len(batches)}: "
                  f"valid={len(result['valid'])}, failed={len(result['failed'])}")
    # Preserve input order and de-duplicate retry list.
    failed_set = set(all_failed)
    ordered_failed = [t for t in tasks if t in failed_set and t not in all_valid]
    return all_valid, ordered_failed, total_cost


def infer_domain_display(original_df, folder_name):
    if "domain" in original_df.columns:
        vals = original_df["domain"].dropna().astype(str).str.strip()
        vals = vals[vals != ""].unique().tolist()
        if len(vals) == 1:
            return vals[0]
    return folder_name.replace("_", " & ") if folder_name == "Education_Research" else folder_name.replace("_", " ")


def save_test_output(domain_out, sampled_df, assignments):
    out = pd.DataFrame({
        "original_task": sampled_df["task"],
        "frequency": sampled_df["frequency"],
        "assigned_task": [assignments.get(t, pd.NA) for t in sampled_df["task"]],
        "correct": "",
        "correct_task": "",
        "notes": "",
    })
    out.to_csv(domain_out / "test_assignments.csv", index=False)
    return out


def save_full_outputs(domain_out, original_df, assignments, attempts, statuses, domain_name):
    mapping = pd.DataFrame({
        "domain": domain_name,
        "original_task": original_df["task"],
        "frequency": original_df["frequency"],
        "assigned_task": [assignments.get(t, pd.NA) for t in original_df["task"]],
        "assignment_status": [statuses.get(t, "failed") for t in original_df["task"]],
        "attempt": [attempts.get(t, 2) for t in original_df["task"]],
    })
    mapping.to_csv(domain_out / "assignments.csv", index=False)

    temp = mapping.copy()
    temp["assigned_task"] = temp["assigned_task"].fillna("NA")
    temp["frequency_numeric"] = pd.to_numeric(temp["frequency"], errors="coerce").fillna(0)
    agg = temp.groupby(["domain", "assigned_task"], as_index=False).agg(
        frequency=("frequency_numeric", "sum"),
        unique_original_tasks=("original_task", "count"),
    )
    agg = agg.rename(columns={"assigned_task": "task"})
    agg.to_csv(domain_out / "domain_tasks_assigned.csv", index=False)
    return mapping, agg


def main():
    ap = argparse.ArgumentParser(description="Assign original Task labels to an extracted domain taxonomy.")
    ap.add_argument("--mode", choices=["test", "full"], required=True)
    ap.add_argument("--taxonomy_run_dir", required=True)
    ap.add_argument("--domain_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--results_dir", default="assignment_results")
    ap.add_argument(
        "--model",
        default="gpt-5.5",
        choices=["gpt-5.5", "gpt-5-mini", "gpt-5-nano"],
    )
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    taxonomy_run_dir = Path(args.taxonomy_run_dir).expanduser().resolve()
    domain_dir = Path(args.domain_dir).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()
    results_root = Path(args.results_dir).expanduser().resolve() / args.mode
    run_dir = next_run_dir(results_root)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    (run_dir / "assignment_prompt.txt").write_text(prompt_text, encoding="utf-8")

    domains = discover_domains(taxonomy_run_dir, domain_dir)
    client = OpenAI()
    api_log = run_dir / "api_log.csv"
    validation_log = run_dir / "validation_flags.csv"
    error_log = run_dir / "errors.log"

    metadata = {
        "created_at": now_iso(), "mode": args.mode, "model": args.model,
        "api": "responses", "temperature": None, "reasoning_effort": "default",
        "taxonomy_run_dir": str(taxonomy_run_dir), "domain_dir": str(domain_dir),
        "prompt": str(prompt_path), "batch_size": args.batch_size,
        "workers": args.workers, "seed": args.seed,
        "sampling": "95% confidence, +/-10% margin of error, p=0.5, finite population correction" if args.mode == "test" else None,
        "domains": [d["folder"] for d in domains],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    grand_cost = 0.0
    combined_mapping = []
    combined_agg = []
    summaries = []

    print(f"Run directory: {run_dir}")
    print(f"Model: {args.model}")
    print("API: Responses")
    print("Temperature: not supplied")
    print("Reasoning effort: not supplied (model default)")
    print(f"Discovered {len(domains)} domain(s): {', '.join(d['folder'] for d in domains)}")

    for d in domains:
        folder = d["folder"]
        domain_out = run_dir / folder
        domain_out.mkdir(parents=True, exist_ok=True)
        taxonomy_df = load_taxonomy(d["taxonomy_path"])
        original_df = load_original_tasks(d["original_path"])
        domain_name = infer_domain_display(original_df, folder)
        taxonomy_df.to_csv(domain_out / "taxonomy_used.csv", index=False)

        print(f"\n{domain_name}: {len(original_df)} original Tasks; {len(taxonomy_df)} predefined Tasks")

        if args.mode == "test":
            n = sample_size_95_10(len(original_df))
            sampled_df = original_df.sample(n=n, random_state=args.seed).reset_index(drop=True)
            sampled_df.to_csv(domain_out / "sampled_original_tasks.csv", index=False)
            tasks = sampled_df["task"].tolist()
            print(f"  test sample size = {n}")
            result = run_batch(
                client, args.model, prompt_text, domain_name, taxonomy_df, tasks,
                1, 1, domain_out, api_log, validation_log, error_log
            )
            # Test mode intentionally makes only one assignment attempt.
            assignments = dict(result["valid"])
            grand_cost += result["cost"]
            for task in result["failed"]:
                log_validation(validation_log, domain_name, 1, 1, task, "test_unassigned_after_single_attempt")
            save_test_output(domain_out, sampled_df, assignments)
            summaries.append({
                "domain": domain_name, "population": len(original_df), "sample_size": n,
                "valid": len(result["valid"]), "other": sum(v == "Other" for v in result["valid"].values()),
                "failed": len(result["failed"]),
            })
        else:
            tasks = original_df["task"].tolist()
            valid1, failed1, cost1 = run_batches_concurrently(
                client, args.model, prompt_text, domain_name, taxonomy_df, tasks, 1,
                args.batch_size, args.workers, domain_out, api_log, validation_log, error_log
            )
            grand_cost += cost1

            valid2 = {}
            failed2 = []
            cost2 = 0.0
            if failed1:
                (domain_out / "retry_tasks.txt").write_text("\n".join(failed1) + "\n", encoding="utf-8")
                print(f"  retrying {len(failed1)} failed Task(s) once...")
                valid2, failed2, cost2 = run_batches_concurrently(
                    client, args.model, prompt_text, domain_name, taxonomy_df, failed1, 2,
                    args.batch_size, args.workers, domain_out, api_log, validation_log, error_log
                )
                grand_cost += cost2

            assignments = dict(valid1)
            assignments.update(valid2)
            attempts = {t: 1 for t in valid1}
            attempts.update({t: 2 for t in valid2})
            statuses = {t: ("other" if a == "Other" else "valid") for t, a in assignments.items()}
            for task in failed2:
                assignments[task] = pd.NA
                attempts[task] = 2
                statuses[task] = "failed"
                log_validation(validation_log, domain_name, 2, 0, task, "failed_after_retry_marked_NA")

            # If attempt 1 had no failures, failed2 is empty; otherwise failed2 is the final NA set.
            final_failed = failed2
            mapping, agg = save_full_outputs(domain_out, original_df, assignments, attempts, statuses, domain_name)
            combined_mapping.append(mapping)
            combined_agg.append(agg)

            accounted = len(mapping)
            expected = len(original_df)
            if accounted != expected or mapping["original_task"].nunique() != expected:
                raise RuntimeError(f"Completeness check failed for {domain_name}: expected {expected}, got {accounted}")

            summary = {
                "domain": domain_name,
                "original_tasks": expected,
                "valid_assignments": int((mapping["assignment_status"] == "valid").sum()),
                "other": int((mapping["assignment_status"] == "other").sum()),
                "na_after_retry": int((mapping["assignment_status"] == "failed").sum()),
                "retried": len(failed1),
                "accounted_for": accounted,
                "complete": accounted == expected,
            }
            summaries.append(summary)
            (domain_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pd.DataFrame(summaries).to_csv(run_dir / "run_summary.csv", index=False)
    if args.mode == "full" and combined_mapping:
        pd.concat(combined_mapping, ignore_index=True).to_csv(run_dir / "all_assignments.csv", index=False)
        pd.concat(combined_agg, ignore_index=True).to_csv(run_dir / "all_domain_tasks_assigned.csv", index=False)

    metadata["finished_at"] = now_iso()
    metadata["estimated_total_cost_usd"] = grand_cost
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nDone. Estimated API cost: ${grand_cost:.6f}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
