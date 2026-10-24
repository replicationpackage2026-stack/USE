import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

MODEL = "gpt-4.1-mini"
BATCH_SIZE = 50
MAX_WORKERS = 5
RANDOM_SEED = 42

SMALL_SAMPLE_SIZE = 96
MEDIUM_MARGIN_OF_ERROR = 0.02
CONFIDENCE_Z = 1.96
EXPECTED_PROPORTION = 0.5

# API/network failures are retried immediately because the model may never
# have returned a usable response at all.
MAX_API_REQUEST_ATTEMPTS = 3

# Hard per-review ceiling across normal processing + reprocessing.
# Example: 3 means each review can appear in at most 3 processing batches.
MAX_REVIEW_ATTEMPTS = 3

MODEL_PRICES = {
    "gpt-4.1-mini": {
        "input_per_1m": 0.4,
        "cached_input_per_1m": 0.10,
        "output_per_1m": 1.6,
    },
}


# ============================================================
# Helpers
# ============================================================

write_lock = Lock()
client = OpenAI()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def load_jsonl(path):
    if not path.exists():
        return []

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping malformed JSONL line "
                    f"{line_number} in {path}"
                )

    return records


def get_review_id(review):
    """
    reviewId is the canonical identifier internally and is sent to the API
    under the simpler key "id".
    """
    return review["reviewId"]


def calculate_cost(model, input_tokens, cached_input_tokens, output_tokens):
    if model not in MODEL_PRICES:
        raise ValueError(f"No pricing information found for model: {model}")

    prices = MODEL_PRICES[model]

    # Cached tokens are already included in total input tokens.
    uncached_input_tokens = max(
        input_tokens - cached_input_tokens,
        0,
    )

    input_cost = (
        uncached_input_tokens
        / 1_000_000
        * prices["input_per_1m"]
    )

    cached_input_cost = (
        cached_input_tokens
        / 1_000_000
        * prices["cached_input_per_1m"]
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * prices["output_per_1m"]
    )

    return {
        "uncached_input_cost_usd": input_cost,
        "cached_input_cost_usd": cached_input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": (
            input_cost
            + cached_input_cost
            + output_cost
        ),
    }


# ============================================================
# Sample size
# ============================================================

def finite_population_sample_size(
    population_size,
    margin_of_error,
    z=CONFIDENCE_Z,
    p=EXPECTED_PROPORTION,
):
    """
    Cochran sample size with finite population correction.
    """
    n0 = (z ** 2 * p * (1 - p)) / (margin_of_error ** 2)
    n = n0 / (1 + ((n0 - 1) / population_size))
    return math.ceil(n)


# ============================================================
# Sample persistence
# ============================================================

def create_or_load_sample(reviews, mode, samples_dir):
    samples_dir.mkdir(parents=True, exist_ok=True)

    if mode == "small":
        sample_path = samples_dir / "small_sample.json"
        target_size = min(SMALL_SAMPLE_SIZE, len(reviews))

    elif mode == "medium":
        sample_path = samples_dir / "medium_sample.json"
        target_size = finite_population_sample_size(
            population_size=len(reviews),
            margin_of_error=MEDIUM_MARGIN_OF_ERROR,
        )
        target_size = min(target_size, len(reviews))

    else:
        return reviews

    if sample_path.exists():
        print(f"Using existing sample: {sample_path}")
        sample = load_json(sample_path)

        input_ids = {get_review_id(r) for r in reviews}
        sample_ids = {get_review_id(r) for r in sample}

        missing = sample_ids - input_ids
        if missing:
            raise ValueError(
                f"The persisted {mode} sample contains "
                f"{len(missing)} reviews not present in the input dataset."
            )

        return sample

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(reviews, target_size)

    save_json(sample, sample_path)

    print(
        f"Created {mode} sample with {len(sample):,} reviews: "
        f"{sample_path}"
    )

    return sample


# ============================================================
# Run directories
# ============================================================

def next_run_directory(output_dir, mode):
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    existing_numbers = []

    for path in mode_dir.glob("run_*"):
        if not path.is_dir():
            continue

        try:
            existing_numbers.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue

    next_number = max(existing_numbers, default=0) + 1

    run_dir = mode_dir / f"run_{next_number:03d}"
    run_dir.mkdir(parents=True)

    return run_dir


def resolve_run_directory(output_dir, mode, resume_run):
    if resume_run:
        run_dir = Path(resume_run).expanduser().resolve()

        if not run_dir.exists():
            raise FileNotFoundError(
                f"Resume directory does not exist: {run_dir}"
            )

        return run_dir

    return next_run_directory(output_dir, mode)


# ============================================================
# Checkpoint handling
# ============================================================

def load_checkpoint_map(checkpoint_path):
    """
    If duplicate IDs somehow appear in the append-only checkpoint,
    the most recently written result wins.
    """
    records = load_jsonl(checkpoint_path)

    checkpoint_map = {}

    for record in records:
        review_id = record.get("reviewId")

        if review_id is not None:
            checkpoint_map[review_id] = record

    return checkpoint_map


# ============================================================
# Persistent per-review processing attempts
# ============================================================

def load_review_attempt_counts(attempt_log_path):
    """
    Reconstruct how many processing batches each review has already been
    submitted in. This persists the retry ceiling across --resume_run.

    Important:
    - One processing-batch submission counts as one review attempt.
    - Immediate API/network retries inside that submission do NOT create
      additional review attempts.
    """
    counts = Counter()

    for record in load_jsonl(attempt_log_path):
        for review_id in record.get("review_ids", []):
            counts[review_id] += 1

    return counts


def record_processing_attempt(
    attempt_log_path,
    processing_round,
    batch_number,
    batch,
):
    """
    Record the attempt BEFORE sending the batch. This makes the retry ceiling
    conservative and resume-safe even if the process crashes mid-request.
    """
    append_jsonl(
        {
            "timestamp": utc_now(),
            "processing_round": processing_round,
            "batch_number": batch_number,
            "review_ids": [
                get_review_id(review)
                for review in batch
            ],
        },
        attempt_log_path,
    )


# ============================================================
# Prompt / API formatting
# ============================================================

def prepare_api_reviews(batch):
    """
    Only review ID and content are sent to the model.
    App/category/rating/etc. are deliberately excluded.
    """
    return [
        {
            "id": get_review_id(review),
            "content": review.get("content", ""),
        }
        for review in batch
    ]


def build_user_message(batch):
    return json.dumps(
        prepare_api_reviews(batch),
        ensure_ascii=False,
        indent=2,
    )


def strip_code_fences(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


# ============================================================
# Validation
# ============================================================

ALLOWED_USE_STATUS = {"actual", "desired", None}


def validate_usage(usage):
    if not isinstance(usage, dict):
        raise ValueError("Usage entry is not an object.")

    required_fields = {
        "domain",
        "task",
        "activity",
        "use_status",
    }

    actual_fields = set(usage.keys())

    missing_fields = required_fields - actual_fields
    extra_fields = actual_fields - required_fields

    if missing_fields:
        raise ValueError(
            f"Usage entry missing fields: {sorted(missing_fields)}"
        )

    if extra_fields:
        raise ValueError(
            f"Usage entry contains unexpected fields: "
            f"{sorted(extra_fields)}"
        )

    # Domain is optional: it may be a non-empty string or null.
    domain = usage["domain"]

    if domain is not None:
        if not isinstance(domain, str):
            raise ValueError(
                "domain must be a string or null."
            )

        if not domain.strip():
            raise ValueError(
                "domain must be a non-empty string or null."
            )

    # Task is required and must be a non-empty string.
    task = usage["task"]

    if not isinstance(task, str) or not task.strip():
        raise ValueError(
            "task must be a non-empty string."
        )

    # Activity is required and must be a non-empty string.
    activity = usage["activity"]

    if not isinstance(activity, str) or not activity.strip():
        raise ValueError(
            "activity must be a non-empty string."
        )

    # Use Status is optional.
    if usage["use_status"] not in ALLOWED_USE_STATUS:
        raise ValueError(
            "use_status must be 'actual', 'desired', or null."
        )


def validate_response_partially(parsed, expected_ids):
    """
    Validate each returned review independently.

    Valid expected reviews are salvaged even if other records in the same
    response are missing, malformed, duplicated, or have unexpected IDs.

    Returns:
        valid_results:
            expected ID -> fully valid record
        missing_ids:
            expected IDs for which no valid result was obtained
        anomalies:
            detailed anomaly records for logging
    """
    if not isinstance(parsed, list):
        raise ValueError(
            "Top-level response must be a JSON array."
        )

    valid_results = {}
    anomalies = []

    for position, record in enumerate(parsed):
        # We cannot associate a non-object result with a specific expected ID.
        if not isinstance(record, dict):
            anomalies.append(
                {
                    "type": "invalid_record",
                    "position": position,
                    "review_id": None,
                    "reason": "Returned review result is not a JSON object.",
                }
            )
            continue

        review_id = record.get("id")

        # Returned object has no usable ID.
        if review_id is None:
            anomalies.append(
                {
                    "type": "invalid_record",
                    "position": position,
                    "review_id": None,
                    "reason": "Returned review result is missing 'id'.",
                }
            )
            continue

        # Model returned an ID that was not included in this batch.
        if review_id not in expected_ids:
            anomalies.append(
                {
                    "type": "unexpected_id",
                    "position": position,
                    "review_id": review_id,
                    "reason": (
                        "Model returned an ID that was not sent "
                        "in this batch."
                    ),
                }
            )
            continue

        required_fields = {"id", "usages"}

        if set(record.keys()) != required_fields:
            missing_fields = required_fields - set(record.keys())
            extra_fields = set(record.keys()) - required_fields

            anomalies.append(
                {
                    "type": "invalid_expected_record",
                    "position": position,
                    "review_id": review_id,
                    "reason": (
                        "Expected review result does not contain exactly "
                        "'id' and 'usages'."
                    ),
                    "missing_fields": sorted(missing_fields),
                    "extra_fields": sorted(extra_fields),
                }
            )
            continue

        usages = record["usages"]

        if not isinstance(usages, list):
            anomalies.append(
                {
                    "type": "invalid_expected_record",
                    "position": position,
                    "review_id": review_id,
                    "reason": "'usages' is not a list.",
                }
            )
            continue

        try:
            for usage in usages:
                validate_usage(usage)
        except ValueError as e:
            anomalies.append(
                {
                    "type": "invalid_expected_record",
                    "position": position,
                    "review_id": review_id,
                    "reason": str(e),
                }
            )
            continue

        # Keep the first fully valid result. Log later duplicates.
        if review_id in valid_results:
            anomalies.append(
                {
                    "type": "duplicate_expected_id",
                    "position": position,
                    "review_id": review_id,
                    "reason": (
                        "A second valid result was returned for the "
                        "same expected review ID. The first valid result "
                        "was retained."
                    ),
                }
            )
            continue

        valid_results[review_id] = record

    missing_ids = expected_ids - set(valid_results.keys())

    return {
        "valid_results": valid_results,
        "missing_ids": missing_ids,
        "anomalies": anomalies,
    }


# ============================================================
# API request
# ============================================================

def request_openai(batch, prompt, batch_number):
    """
    Retry only API/request-level failures immediately.

    A malformed or partially invalid MODEL OUTPUT is not retried here.
    It is returned to the processing layer, where valid reviews are salvaged
    and unresolved reviews are pooled for a later reprocessing round.
    """
    last_error = None

    for api_attempt in range(1, MAX_API_REQUEST_ATTEMPTS + 1):
        start_time = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": build_user_message(batch),
                    },
                ],
                temperature=0,
            )

            elapsed = time.perf_counter() - start_time

            return {
                "response": response,
                "api_attempts": api_attempt,
                "runtime_seconds": elapsed,
                "error": None,
            }

        except Exception as e:
            last_error = str(e)

            if api_attempt < MAX_API_REQUEST_ATTEMPTS:
                sleep_seconds = min(2 ** api_attempt, 10)

                print(
                    f"Batch {batch_number} API attempt {api_attempt} "
                    f"failed: {last_error}. "
                    f"Retrying in {sleep_seconds}s..."
                )

                time.sleep(sleep_seconds)

    return {
        "response": None,
        "api_attempts": MAX_API_REQUEST_ATTEMPTS,
        "runtime_seconds": None,
        "error": last_error,
    }


def get_usage_stats(response):
    usage = response.usage

    input_tokens = getattr(
        usage,
        "prompt_tokens",
        0,
    ) or 0

    output_tokens = getattr(
        usage,
        "completion_tokens",
        0,
    ) or 0

    total_tokens = getattr(
        usage,
        "total_tokens",
        input_tokens + output_tokens,
    ) or (input_tokens + output_tokens)

    cached_input_tokens = 0

    prompt_details = getattr(
        usage,
        "prompt_tokens_details",
        None,
    )

    if prompt_details is not None:
        cached_input_tokens = getattr(
            prompt_details,
            "cached_tokens",
            0,
        ) or 0

    costs = calculate_cost(
        model=MODEL,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "uncached_input_cost_usd": round(
            costs["uncached_input_cost_usd"],
            8,
        ),
        "cached_input_cost_usd": round(
            costs["cached_input_cost_usd"],
            8,
        ),
        "output_cost_usd": round(
            costs["output_cost_usd"],
            8,
        ),
        "total_cost_usd": round(
            costs["total_cost_usd"],
            8,
        ),
    }


def call_batch(
    batch,
    prompt,
    batch_number,
    processing_round,
):
    """
    One processing attempt for every review in `batch`.

    Outcomes:
    - valid expected outputs are returned immediately for checkpointing
    - unexpected IDs are logged and ignored
    - malformed expected outputs are logged and ignored
    - any expected review without a valid result becomes missing and is
      eligible for a later pooled retry if under MAX_REVIEW_ATTEMPTS
    """
    started_at = utc_now()

    expected_ids = {
        get_review_id(review)
        for review in batch
    }

    request_result = request_openai(
        batch=batch,
        prompt=prompt,
        batch_number=batch_number,
    )

    response = request_result["response"]

    # --------------------------------------------------------
    # API/request failed after all immediate retries
    # --------------------------------------------------------
    if response is None:
        return {
            "results": {},
            "missing_ids": expected_ids,
            "anomalies": [],
            "raw_response": None,
            "log": {
                "timestamp": started_at,
                "processing_round": processing_round,
                "batch_number": batch_number,
                "model": MODEL,
                "requested_reviews": len(batch),
                "returned_valid_reviews": 0,
                "missing_reviews": len(batch),
                "unexpected_ids": 0,
                "invalid_expected_records": 0,
                "duplicate_expected_ids": 0,
                "api_attempts": request_result["api_attempts"],
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "uncached_input_cost_usd": 0,
                "cached_input_cost_usd": 0,
                "output_cost_usd": 0,
                "total_cost_usd": 0,
                "runtime_seconds": None,
                "status": "request_failed",
                "error": request_result["error"],
            },
        }

    raw_text = response.choices[0].message.content or ""
    raw_text = strip_code_fences(raw_text)

    token_stats = get_usage_stats(response)

    # --------------------------------------------------------
    # Top-level JSON cannot be parsed
    # --------------------------------------------------------
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {
            "results": {},
            "missing_ids": expected_ids,
            "anomalies": [
                {
                    "type": "invalid_top_level_json",
                    "position": None,
                    "review_id": None,
                    "reason": str(e),
                }
            ],
            "raw_response": raw_text,
            "log": {
                "timestamp": started_at,
                "processing_round": processing_round,
                "batch_number": batch_number,
                "model": MODEL,
                "requested_reviews": len(batch),
                "returned_valid_reviews": 0,
                "missing_reviews": len(batch),
                "unexpected_ids": 0,
                "invalid_expected_records": 0,
                "duplicate_expected_ids": 0,
                "api_attempts": request_result["api_attempts"],
                **token_stats,
                "runtime_seconds": round(
                    request_result["runtime_seconds"],
                    3,
                ),
                "status": "invalid_json",
                "error": str(e),
            },
        }

    # --------------------------------------------------------
    # Top-level JSON parsed, but wrong top-level type
    # --------------------------------------------------------
    try:
        validation = validate_response_partially(
            parsed=parsed,
            expected_ids=expected_ids,
        )
    except ValueError as e:
        return {
            "results": {},
            "missing_ids": expected_ids,
            "anomalies": [
                {
                    "type": "invalid_top_level_structure",
                    "position": None,
                    "review_id": None,
                    "reason": str(e),
                }
            ],
            "raw_response": raw_text,
            "log": {
                "timestamp": started_at,
                "processing_round": processing_round,
                "batch_number": batch_number,
                "model": MODEL,
                "requested_reviews": len(batch),
                "returned_valid_reviews": 0,
                "missing_reviews": len(batch),
                "unexpected_ids": 0,
                "invalid_expected_records": 0,
                "duplicate_expected_ids": 0,
                "api_attempts": request_result["api_attempts"],
                **token_stats,
                "runtime_seconds": round(
                    request_result["runtime_seconds"],
                    3,
                ),
                "status": "invalid_structure",
                "error": str(e),
            },
        }

    anomalies = validation["anomalies"]

    anomaly_counts = Counter(
        anomaly["type"]
        for anomaly in anomalies
    )

    missing_ids = validation["missing_ids"]
    valid_results = validation["valid_results"]

    if not missing_ids and not anomalies:
        status = "success"
    elif valid_results:
        status = "partial"
    else:
        status = "no_valid_results"

    log_record = {
        "timestamp": started_at,
        "processing_round": processing_round,
        "batch_number": batch_number,
        "model": MODEL,
        "requested_reviews": len(batch),
        "returned_valid_reviews": len(valid_results),
        "missing_reviews": len(missing_ids),
        "unexpected_ids": anomaly_counts.get(
            "unexpected_id",
            0,
        ),
        "invalid_expected_records": anomaly_counts.get(
            "invalid_expected_record",
            0,
        ),
        "duplicate_expected_ids": anomaly_counts.get(
            "duplicate_expected_id",
            0,
        ),
        "other_anomalies": (
            len(anomalies)
            - anomaly_counts.get("unexpected_id", 0)
            - anomaly_counts.get("invalid_expected_record", 0)
            - anomaly_counts.get("duplicate_expected_id", 0)
        ),
        "api_attempts": request_result["api_attempts"],
        **token_stats,
        "runtime_seconds": round(
            request_result["runtime_seconds"],
            3,
        ),
        "status": status,
        "error": None,
    }

    return {
        "results": valid_results,
        "missing_ids": missing_ids,
        "anomalies": anomalies,
        "raw_response": raw_text,
        "log": log_record,
    }


# ============================================================
# Store successful results
# ============================================================

def checkpoint_successes(
    batch,
    returned_map,
    checkpoint_path,
):
    """
    Store original metadata + extracted usages.

    A returned review with usages=[] is a valid completed review.
    """
    original_map = {
        get_review_id(review): review
        for review in batch
    }

    completed = []

    for review_id, result in returned_map.items():
        original = original_map[review_id]

        checkpoint_record = dict(original)
        checkpoint_record["usages"] = result["usages"]

        append_jsonl(
            checkpoint_record,
            checkpoint_path,
        )

        completed.append(review_id)

    return completed


# ============================================================
# Anomaly / retry logging
# ============================================================

def log_anomalies(
    anomalies,
    anomaly_path,
    processing_round,
    batch_number,
):
    for anomaly in anomalies:
        append_jsonl(
            {
                "timestamp": utc_now(),
                "processing_round": processing_round,
                "batch_number": batch_number,
                **anomaly,
            },
            anomaly_path,
        )


def log_missing_reviews(
    missing_ids,
    retry_path,
    attempt_counts,
    processing_round,
    batch_number,
):
    if not missing_ids:
        return

    append_jsonl(
        {
            "timestamp": utc_now(),
            "processing_round": processing_round,
            "batch_number": batch_number,
            "review_ids": sorted(missing_ids),
            "attempts_after_this_batch": {
                review_id: attempt_counts.get(review_id, 0)
                for review_id in sorted(missing_ids)
            },
        },
        retry_path,
    )


# ============================================================
# Batch processing
# ============================================================

def chunk_reviews(reviews, size):
    for i in range(0, len(reviews), size):
        yield reviews[i:i + size]


def order_retry_reviews(
    eligible_reviews,
    processing_round,
):
    """
    Round 1 preserves sample/dataset order.

    Later rounds deterministically shuffle unresolved reviews so failures from
    different original batches are mixed together when possible. This avoids
    repeatedly recreating the same batch composition while remaining
    reproducible.

    The last retry batch may be smaller than BATCH_SIZE, including a single
    review if only one unresolved review remains.
    """
    ordered = list(eligible_reviews)

    if processing_round > 1:
        rng = random.Random(
            RANDOM_SEED + processing_round
        )
        rng.shuffle(ordered)

    return ordered


def handle_batch_result(
    batch,
    result,
    checkpoint_path,
    request_log_path,
    anomaly_path,
    retry_path,
    attempt_counts,
    processing_round,
    batch_number,
):
    append_jsonl(
        result["log"],
        request_log_path,
    )

    completed = checkpoint_successes(
        batch=batch,
        returned_map=result["results"],
        checkpoint_path=checkpoint_path,
    )

    log_anomalies(
        anomalies=result["anomalies"],
        anomaly_path=anomaly_path,
        processing_round=processing_round,
        batch_number=batch_number,
    )

    log_missing_reviews(
        missing_ids=result["missing_ids"],
        retry_path=retry_path,
        attempt_counts=attempt_counts,
        processing_round=processing_round,
        batch_number=batch_number,
    )

    return completed


def process_reviews(
    reviews,
    prompt,
    run_dir,
    concurrent,
):
    checkpoint_path = run_dir / "checkpoint.jsonl"
    request_log_path = run_dir / "request_log.jsonl"
    attempt_log_path = run_dir / "processing_attempts.jsonl"
    anomaly_path = run_dir / "response_anomalies.jsonl"
    retry_path = run_dir / "retry_log.jsonl"
    final_failure_path = run_dir / "final_failures.jsonl"

    # Do not keep stale final-failure output when resuming.
    if final_failure_path.exists():
        final_failure_path.unlink()

    review_map = {
        get_review_id(review): review
        for review in reviews
    }

    target_ids = set(review_map.keys())

    checkpoint_map = load_checkpoint_map(
        checkpoint_path
    )

    completed_ids = (
        set(checkpoint_map.keys())
        & target_ids
    )

    attempt_counts = load_review_attempt_counts(
        attempt_log_path
    )

    print(f"Target reviews: {len(target_ids):,}")
    print(f"Already completed: {len(completed_ids):,}")

    batch_counter = 0
    processing_round = 0

    while True:
        checkpoint_map = load_checkpoint_map(
            checkpoint_path
        )

        completed_ids = (
            set(checkpoint_map.keys())
            & target_ids
        )

        unresolved_ids = target_ids - completed_ids

        if not unresolved_ids:
            print("All target reviews have been processed.")
            break

        eligible_ids = {
            review_id
            for review_id in unresolved_ids
            if attempt_counts.get(review_id, 0)
            < MAX_REVIEW_ATTEMPTS
        }

        exhausted_ids = (
            unresolved_ids - eligible_ids
        )

        if not eligible_ids:
            print()
            print(
                f"No unresolved reviews remain eligible for retry. "
                f"{len(exhausted_ids):,} reached the hard limit of "
                f"{MAX_REVIEW_ATTEMPTS} processing attempts."
            )
            break

        processing_round += 1

        # Defensive hard stop. In normal operation this can never exceed the
        # per-review maximum because every eligible review gains one attempt
        # each round it is submitted.
        if processing_round > MAX_REVIEW_ATTEMPTS + 1:
            raise RuntimeError(
                "Defensive processing-round limit reached unexpectedly."
            )

        # Preserve target-review order before optional retry shuffle.
        eligible_reviews = [
            review
            for review in reviews
            if get_review_id(review) in eligible_ids
        ]

        eligible_reviews = order_retry_reviews(
            eligible_reviews=eligible_reviews,
            processing_round=processing_round,
        )

        print()
        if processing_round == 1:
            print(
                f"Processing round {processing_round}: "
                f"{len(eligible_reviews):,} reviews."
            )
        else:
            print(
                f"Reprocessing round {processing_round - 1}: "
                f"{len(eligible_reviews):,} unresolved reviews "
                f"pooled and rebatched."
            )

        batches = list(
            chunk_reviews(
                eligible_reviews,
                BATCH_SIZE,
            )
        )

        # ----------------------------------------------------
        # Concurrent mode
        # ----------------------------------------------------
        if concurrent:
            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                future_map = {}
                batch_iterator = iter(batches)

                def submit_next_batch():
                    nonlocal batch_counter

                    try:
                        batch = next(batch_iterator)
                    except StopIteration:
                        return False

                    batch_counter += 1
                    current_batch_number = batch_counter

                    # Record the attempt immediately before submitting
                    # this batch to the executor.
                    record_processing_attempt(
                        attempt_log_path=attempt_log_path,
                        processing_round=processing_round,
                        batch_number=current_batch_number,
                        batch=batch,
                    )

                    for review in batch:
                        attempt_counts[
                            get_review_id(review)
                        ] += 1

                    future = executor.submit(
                        call_batch,
                        batch,
                        prompt,
                        current_batch_number,
                        processing_round,
                    )

                    future_map[future] = (
                        current_batch_number,
                        batch,
                    )

                    return True

                # Initially submit only enough batches to occupy
                # the available workers.
                for _ in range(MAX_WORKERS):
                    if not submit_next_batch():
                        break

                # As each batch finishes, process its result and
                # submit exactly one replacement batch.
                while future_map:
                    done_future = next(
                        as_completed(future_map)
                    )

                    current_batch_number, batch = (
                        future_map.pop(done_future)
                    )

                    try:
                        result = done_future.result()

                    except Exception as e:
                        # Defensive fallback for unexpected local exceptions.
                        expected_ids = {
                            get_review_id(review)
                            for review in batch
                        }

                        result = {
                            "results": {},
                            "missing_ids": expected_ids,
                            "anomalies": [
                                {
                                    "type": "local_processing_error",
                                    "position": None,
                                    "review_id": None,
                                    "reason": str(e),
                                }
                            ],
                            "raw_response": None,
                            "log": {
                                "timestamp": utc_now(),
                                "processing_round": processing_round,
                                "batch_number": current_batch_number,
                                "model": MODEL,
                                "requested_reviews": len(batch),
                                "returned_valid_reviews": 0,
                                "missing_reviews": len(batch),
                                "unexpected_ids": 0,
                                "invalid_expected_records": 0,
                                "duplicate_expected_ids": 0,
                                "api_attempts": 0,
                                "input_tokens": 0,
                                "cached_input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                                "uncached_input_cost_usd": 0,
                                "cached_input_cost_usd": 0,
                                "output_cost_usd": 0,
                                "total_cost_usd": 0,
                                "runtime_seconds": None,
                                "status": "local_processing_error",
                                "error": str(e),
                            },
                        }

                    completed = handle_batch_result(
                        batch=batch,
                        result=result,
                        checkpoint_path=checkpoint_path,
                        request_log_path=request_log_path,
                        anomaly_path=anomaly_path,
                        retry_path=retry_path,
                        attempt_counts=attempt_counts,
                        processing_round=processing_round,
                        batch_number=current_batch_number,
                    )

                    print(
                        f"Batch {current_batch_number}: "
                        f"{len(completed)}/{len(batch)} completed"
                    )

                    # One worker slot has become free, so submit
                    # exactly one new batch.
                    submit_next_batch()
        # ----------------------------------------------------
        # Sequential mode
        # ----------------------------------------------------
        else:
            for batch in batches:
                batch_counter += 1

                record_processing_attempt(
                    attempt_log_path=attempt_log_path,
                    processing_round=processing_round,
                    batch_number=batch_counter,
                    batch=batch,
                )

                for review in batch:
                    attempt_counts[
                        get_review_id(review)
                    ] += 1

                result = call_batch(
                    batch=batch,
                    prompt=prompt,
                    batch_number=batch_counter,
                    processing_round=processing_round,
                )

                completed = handle_batch_result(
                    batch=batch,
                    result=result,
                    checkpoint_path=checkpoint_path,
                    request_log_path=request_log_path,
                    anomaly_path=anomaly_path,
                    retry_path=retry_path,
                    attempt_counts=attempt_counts,
                    processing_round=processing_round,
                    batch_number=batch_counter,
                )

                print(
                    f"Batch {batch_counter}: "
                    f"{len(completed)}/{len(batch)} completed"
                )

        # The loop now recomputes unresolved reviews from checkpoint.jsonl.
        # Only reviews with no valid checkpoint and remaining attempts enter
        # the next pooled retry round.

    # --------------------------------------------------------
    # Final completeness / hard-limit check
    # --------------------------------------------------------
    checkpoint_map = load_checkpoint_map(
        checkpoint_path
    )

    completed_ids = (
        set(checkpoint_map.keys())
        & target_ids
    )

    remaining_ids = target_ids - completed_ids

    if remaining_ids:
        # Write one terminal record per unresolved review.
        for review in reviews:
            review_id = get_review_id(review)

            if review_id not in remaining_ids:
                continue

            failure_record = dict(review)
            failure_record["processing_attempts"] = (
                attempt_counts.get(review_id, 0)
            )
            failure_record["failure_reason"] = (
                "No valid result was obtained before reaching "
                "the retry limit."
            )

            append_jsonl(
                failure_record,
                final_failure_path,
            )

    return remaining_ids


# ============================================================
# Final JSONL
# ============================================================

def compile_final_jsonl(
    reviews,
    checkpoint_path,
    output_path,
):
    checkpoint_map = load_checkpoint_map(
        checkpoint_path
    )

    expected_ids = [
        get_review_id(review)
        for review in reviews
    ]

    expected_set = set(expected_ids)
    completed_set = set(checkpoint_map.keys())

    missing = expected_set - completed_set

    if missing:
        raise RuntimeError(
            f"Cannot compile final dataset. "
            f"{len(missing)} reviews are still missing."
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for review in reviews:
            review_id = get_review_id(review)
            record = checkpoint_map[review_id]

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Final JSONL compiled: {output_path}")


# ============================================================
# Run summary
# ============================================================

def create_run_summary(
    run_dir,
    mode,
    reviews,
    remaining_ids,
    start_time,
):
    logs = load_jsonl(
        run_dir / "request_log.jsonl"
    )

    anomalies = load_jsonl(
        run_dir / "response_anomalies.jsonl"
    )

    attempt_counts = load_review_attempt_counts(
        run_dir / "processing_attempts.jsonl"
    )

    checkpoint_map = load_checkpoint_map(
        run_dir / "checkpoint.jsonl"
    )

    target_ids = {
        get_review_id(review)
        for review in reviews
    }

    input_tokens = sum(
        log.get("input_tokens", 0) or 0
        for log in logs
    )

    cached_input_tokens = sum(
        log.get("cached_input_tokens", 0) or 0
        for log in logs
    )

    output_tokens = sum(
        log.get("output_tokens", 0) or 0
        for log in logs
    )

    total_tokens = sum(
        log.get("total_tokens", 0) or 0
        for log in logs
    )

    statuses = Counter(
        log.get("status", "unknown")
        for log in logs
    )

    anomaly_types = Counter(
        record.get("type", "unknown")
        for record in anomalies
    )

    total_usages = sum(
        len(record.get("usages", []))
        for review_id, record in checkpoint_map.items()
        if review_id in target_ids
    )

    zero_usage_reviews = sum(
        1
        for review_id, record in checkpoint_map.items()
        if review_id in target_ids
        and record.get("usages") == []
    )

    uncached_input_cost_usd = sum(
        log.get("uncached_input_cost_usd", 0) or 0
        for log in logs
    )

    cached_input_cost_usd = sum(
        log.get("cached_input_cost_usd", 0) or 0
        for log in logs
    )

    output_cost_usd = sum(
        log.get("output_cost_usd", 0) or 0
        for log in logs
    )

    total_cost_usd = sum(
        log.get("total_cost_usd", 0) or 0
        for log in logs
    )

    attempted_target_ids = {
        review_id
        for review_id in target_ids
        if attempt_counts.get(review_id, 0) > 0
    }

    reviews_reprocessed = sum(
        1
        for review_id in target_ids
        if attempt_counts.get(review_id, 0) > 1
    )

    max_attempts_observed = max(
        (
            attempt_counts.get(review_id, 0)
            for review_id in target_ids
        ),
        default=0,
    )

    summary = {
        "mode": mode,
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "max_workers": MAX_WORKERS,
        "random_seed": RANDOM_SEED,
        "max_api_request_attempts":
            MAX_API_REQUEST_ATTEMPTS,
        "max_review_attempts":
            MAX_REVIEW_ATTEMPTS,
        "target_reviews": len(reviews),
        "attempted_reviews": len(
            attempted_target_ids
        ),
        "completed_reviews": len(
            set(checkpoint_map.keys())
            & target_ids
        ),
        "remaining_reviews": len(
            remaining_ids
        ),
        "reviews_reprocessed":
            reviews_reprocessed,
        "max_review_attempts_observed":
            max_attempts_observed,
        "total_extracted_usages":
            total_usages,
        "reviews_with_zero_usages":
            zero_usage_reviews,
        "api_requests_logged": len(logs),
        "request_status_counts":
            dict(statuses),
        "response_anomaly_counts":
            dict(anomaly_types),
        "input_tokens": input_tokens,
        "cached_input_tokens":
            cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "uncached_input_cost_usd": round(
            uncached_input_cost_usd,
            6,
        ),
        "cached_input_cost_usd": round(
            cached_input_cost_usd,
            6,
        ),
        "output_cost_usd": round(
            output_cost_usd,
            6,
        ),
        "total_cost_usd": round(
            total_cost_usd,
            6,
        ),
        "wall_clock_seconds": round(
            time.perf_counter() - start_time,
            3,
        ),
        "finished_at": utc_now(),
    }

    save_json(
        summary,
        run_dir / "run_summary.json",
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract structured usage dimensions "
            "from Gen-AI application reviews."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["small", "medium", "full"],
        help=(
            "small = fixed 96-review evaluation sample; "
            "medium = 95%% confidence / 2%% MOE sample; "
            "full = complete dataset"
        ),
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSON file containing filtered reviews.",
    )

    parser.add_argument(
        "--prompt_file",
        required=True,
        help="Text file containing the extraction prompt.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base directory for samples and run outputs.",
    )

    parser.add_argument(
        "--resume_run",
        default=None,
        help=(
            "Existing run directory to resume. "
            "Example: results/full/run_001"
        ),
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    input_file = Path(
        args.input_file
    ).expanduser().resolve()

    prompt_file = Path(
        args.prompt_file
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------
    reviews = load_json(input_file)

    if not isinstance(reviews, list):
        raise ValueError(
            "Input JSON must contain a list of review objects."
        )

    review_ids = [
        get_review_id(review)
        for review in reviews
    ]

    if len(review_ids) != len(set(review_ids)):
        raise ValueError(
            "Input contains duplicate reviewId values."
        )

    print(f"Loaded {len(reviews):,} reviews.")

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------
    with open(
        prompt_file,
        "r",
        encoding="utf-8",
    ) as f:
        prompt = f.read().strip()

    # --------------------------------------------------------
    # Select sample/full dataset
    # --------------------------------------------------------
    samples_dir = output_dir / "samples"

    target_reviews = create_or_load_sample(
        reviews=reviews,
        mode=args.mode,
        samples_dir=samples_dir,
    )

    print(f"Mode: {args.mode}")
    print(
        f"Reviews for this mode: "
        f"{len(target_reviews):,}"
    )

    # --------------------------------------------------------
    # Run directory
    # --------------------------------------------------------
    run_dir = resolve_run_directory(
        output_dir=output_dir,
        mode=args.mode,
        resume_run=args.resume_run,
    )

    print(f"Run directory: {run_dir}")

    # --------------------------------------------------------
    # New run vs resume
    # --------------------------------------------------------
    run_config_path = (
        run_dir / "run_config.json"
    )

    saved_prompt_path = (
        run_dir / "prompt.txt"
    )

    if args.resume_run:
        if not run_config_path.exists():
            raise FileNotFoundError(
                "Cannot resume: run_config.json is missing."
            )

        config = load_json(
            run_config_path
        )

        if config.get("mode") != args.mode:
            raise ValueError(
                "Resume mode does not match the original run."
            )

        if config.get("model") != MODEL:
            raise ValueError(
                "Resume model does not match the original run."
            )

        if (
            config.get("max_review_attempts")
            is not None
            and config.get("max_review_attempts")
            != MAX_REVIEW_ATTEMPTS
        ):
            raise ValueError(
                "MAX_REVIEW_ATTEMPTS differs from the original run. "
                "Resume aborted."
            )

        if saved_prompt_path.exists():
            with open(
                saved_prompt_path,
                "r",
                encoding="utf-8",
            ) as f:
                original_prompt = (
                    f.read().strip()
                )

            if original_prompt != prompt:
                raise ValueError(
                    "The supplied prompt differs from the prompt used by "
                    "the original run. Resume aborted."
                )

        print("Resuming existing run.")

    else:
        config = {
            "created_at": utc_now(),
            "mode": args.mode,
            "model": MODEL,
            "batch_size": BATCH_SIZE,
            "max_workers": MAX_WORKERS,
            "random_seed": RANDOM_SEED,
            "input_file": str(input_file),
            "prompt_file": str(prompt_file),
            "target_review_count":
                len(target_reviews),
            "small_sample_size":
                SMALL_SAMPLE_SIZE,
            "medium_margin_of_error":
                MEDIUM_MARGIN_OF_ERROR,
            "confidence_level": 0.95,
            "max_api_request_attempts":
                MAX_API_REQUEST_ATTEMPTS,
            "max_review_attempts":
                MAX_REVIEW_ATTEMPTS,
        }

        save_json(
            config,
            run_config_path,
        )

        with open(
            saved_prompt_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(prompt)

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------
    concurrent = (
        args.mode in {"medium", "full"}
    )

    remaining_ids = process_reviews(
        reviews=target_reviews,
        prompt=prompt,
        run_dir=run_dir,
        concurrent=concurrent,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary = create_run_summary(
        run_dir=run_dir,
        mode=args.mode,
        reviews=target_reviews,
        remaining_ids=remaining_ids,
        start_time=start_time,
    )

    print()
    print("=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)

    print(
        f"Completed: "
        f"{summary['completed_reviews']:,}/"
        f"{summary['target_reviews']:,}"
    )

    print(
        f"Reprocessed reviews: "
        f"{summary['reviews_reprocessed']:,}"
    )

    print(
        f"Extracted usages: "
        f"{summary['total_extracted_usages']:,}"
    )

    print(
        f"Zero-usage reviews: "
        f"{summary['reviews_with_zero_usages']:,}"
    )

    print(
        f"Input tokens: "
        f"{summary['input_tokens']:,}"
    )

    print(
        f"Output tokens: "
        f"{summary['output_tokens']:,}"
    )

    print(
        f"Total tokens: "
        f"{summary['total_tokens']:,}"
    )

    print(
        f"Estimated cost: "
        f"${summary['total_cost_usd']:.4f} USD"
    )

    # --------------------------------------------------------
    # Compile final output only when complete
    # --------------------------------------------------------
    if remaining_ids:
        print()
        print(
            f"WARNING: {len(remaining_ids):,} reviews remain "
            f"unprocessed after the hard limit of "
            f"{MAX_REVIEW_ATTEMPTS} processing attempts."
        )

        remaining_path = (
            run_dir
            / "remaining_review_ids.json"
        )

        save_json(
            sorted(remaining_ids),
            remaining_path,
        )

        print(
            f"Remaining IDs saved to: "
            f"{remaining_path}"
        )

        print(
            "Terminal failed reviews saved to: "
            f"{run_dir / 'final_failures.jsonl'}"
        )

        print(
            "These reviews will NOT be retried automatically "
            "on resume because their persisted attempt counts "
            "have reached the configured limit."
        )

    else:
        remaining_path = (
            run_dir
            / "remaining_review_ids.json"
        )

        if remaining_path.exists():
            remaining_path.unlink()

        final_output = (
            run_dir
            / "reviews_with_usage_dimensions.jsonl"
        )

        compile_final_jsonl(
            reviews=target_reviews,
            checkpoint_path=(
                run_dir
                / "checkpoint.jsonl"
            ),
            output_path=final_output,
        )

        print()
        print("Run completed successfully.")


if __name__ == "__main__":
    main()
