import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI


# ============================================================
# Fixed replication settings
# ============================================================

MODEL = "gpt-4.1-mini"
BATCH_SIZE = 100
N_EXAMPLES = 3

MODEL_PRICES = {
    "gpt-4.1-mini": {
        "input_per_1m": 0.4,
        "cached_input_per_1m": 0.10,
        "output_per_1m": 1.6,
    },
}


client = OpenAI()


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter reviews that discuss usage scenarios using an OpenAI model."
    )

    parser.add_argument(
        "--input_file",
        required=True,
        type=Path,
        help="Path to the input JSON file containing reviews.",
    )

    parser.add_argument(
        "--prompt_file",
        required=True,
        type=Path,
        help="Path to the prompt template text file.",
    )

    parser.add_argument(
        "--few_shot_file",
        required=True,
        type=Path,
        help="Path to the few-shot examples JSON file.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where output files will be saved.",
    )

    parser.add_argument(
        "--max_workers",
        required=True,
        type=int,
        help="Number of parallel workers to use.",
    )

    return parser.parse_args()


# ============================================================
# Basic helpers
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def normalize_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.strip().lower()
        if value == "true":
            return True
        if value == "false":
            return False

    return None


def chunk_list(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start // batch_size + 1, items[start:start + batch_size]


def extract_response_text(response):
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks = []

    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)

    return "\n".join(chunks).strip()


def extract_usage(response):
    usage = getattr(response, "usage", None)

    if usage is None:
        return 0, 0, 0

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    cached_input_tokens = 0
    input_details = getattr(usage, "input_tokens_details", None)

    if input_details is not None:
        cached_input_tokens = getattr(input_details, "cached_tokens", 0) or 0

    return input_tokens, cached_input_tokens, output_tokens


def calculate_cost(model, input_tokens, cached_input_tokens, output_tokens):
    prices = MODEL_PRICES[model]

    non_cached_input_tokens = max(input_tokens - cached_input_tokens, 0)

    input_cost = (non_cached_input_tokens / 1_000_000) * prices["input_per_1m"]
    cached_input_cost = (cached_input_tokens / 1_000_000) * prices["cached_input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * prices["output_per_1m"]

    total_cost = input_cost + cached_input_cost + output_cost

    return input_cost, cached_input_cost, output_cost, total_cost


def extract_json_array(raw_text):
    text = raw_text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)

    if not match:
        raise ValueError("Could not find a JSON array in model response.")

    return json.loads(match.group(0))


# ============================================================
# Prompt construction
# ============================================================

def format_few_shot_examples(examples, n_examples):
    selected = examples[:n_examples]

    if not selected:
        return ""

    lines = ["Few-shot examples:"]

    for ex in selected:
        lines.append("\nReview:")
        lines.append(f'{ex["unique_id"]}: {ex["content"]}')
        lines.append("Correct output:")
        lines.append(
            json.dumps(
                [
                    {
                        "unique_id": ex["unique_id"],
                        "use_case": normalize_bool(ex["use_case"]),
                        "reason": ex["reason"],
                    }
                ],
                ensure_ascii=False,
            )
        )

    return "\n".join(lines)


def format_reviews_for_prompt(reviews):
    lines = ["Reviews to classify:"]

    for review in reviews:
        lines.append(f'{review["unique_id"]}: {review.get("content", "")}')

    return "\n".join(lines)


def build_prompt(reviews, prompt_template, few_shot_examples):
    few_shot_text = format_few_shot_examples(few_shot_examples, N_EXAMPLES)
    reviews_text = format_reviews_for_prompt(reviews)

    parts = [prompt_template]

    if few_shot_text:
        parts.append(few_shot_text)

    parts.append(reviews_text)

    return "\n\n".join(parts)


# ============================================================
# OpenAI call
# ============================================================

def call_model(prompt):
    return client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0,
    )


# ============================================================
# Resume support and output writing
# ============================================================

def load_completed_unique_ids(processed_csv_file):
    if not processed_csv_file.exists():
        return set()

    completed = set()

    with open(processed_csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("unique_id"):
                completed.add(row["unique_id"])

    return completed


def append_rows_to_processed_csv(rows, processed_csv_file):
    fieldnames = [
        "unique_id",
        "content",
        "predicted_use_case",
        "reason",
    ]

    file_exists = processed_csv_file.exists()

    with open(processed_csv_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def append_log_row(row, log_file):
    fieldnames = [
        "timestamp",
        "batch_number",
        "model",
        "n_examples",
        "batch_size",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "input_cost",
        "cached_input_cost",
        "output_cost",
        "total_cost",
        "runtime_seconds",
        "status",
        "error_message",
    ]

    file_exists = log_file.exists()

    with open(log_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Batch processing
# ============================================================

def process_batch(
    batch_number,
    reviews,
    prompt_template,
    few_shot_examples,
    first_prompt_file,
    save_first_prompt=False,
):
    prompt = build_prompt(
        reviews=reviews,
        prompt_template=prompt_template,
        few_shot_examples=few_shot_examples,
    )

    if save_first_prompt and not first_prompt_file.exists():
        with open(first_prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

    start = time.perf_counter()

    try:
        response = call_model(prompt)
        runtime_seconds = time.perf_counter() - start

        raw_text = extract_response_text(response)
        parsed = extract_json_array(raw_text)

        input_tokens, cached_input_tokens, output_tokens = extract_usage(response)

        input_cost, cached_input_cost, output_cost, total_cost = calculate_cost(
            model=MODEL,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )

        predictions = {}

        for item in parsed:
            unique_id = item.get("unique_id")
            predictions[unique_id] = {
                "predicted_use_case": normalize_bool(item.get("use_case")),
                "reason": item.get("reason", ""),
            }

        processed_rows = []

        for review in reviews:
            unique_id = review["unique_id"]
            prediction = predictions.get(unique_id, {})

            processed_rows.append(
                {
                    "unique_id": unique_id,
                    "content": review.get("content", ""),
                    "predicted_use_case": prediction.get("predicted_use_case"),
                    "reason": prediction.get("reason", ""),
                }
            )

        return {
            "status": "success",
            "batch_number": batch_number,
            "processed_rows": processed_rows,
            "log_row": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "batch_number": batch_number,
                "model": MODEL,
                "n_examples": N_EXAMPLES,
                "batch_size": len(reviews),
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "cached_input_cost": cached_input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "runtime_seconds": runtime_seconds,
                "status": "success",
                "error_message": "",
            },
        }

    except Exception as e:
        runtime_seconds = time.perf_counter() - start

        return {
            "status": "error",
            "batch_number": batch_number,
            "processed_rows": [],
            "log_row": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "batch_number": batch_number,
                "model": MODEL,
                "n_examples": N_EXAMPLES,
                "batch_size": len(reviews),
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0,
                "cached_input_cost": 0,
                "output_cost": 0,
                "total_cost": 0,
                "runtime_seconds": runtime_seconds,
                "status": "error",
                "error_message": str(e),
            },
        }


# ============================================================
# Final JSON export
# ============================================================

def create_true_only_json(original_reviews, processed_csv_file, final_true_json_file):
    if not processed_csv_file.exists():
        print("No processed CSV found. Skipping final JSON export.")
        return

    true_ids = set()

    with open(processed_csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if normalize_bool(row.get("predicted_use_case")) is True:
                true_ids.add(row["unique_id"])

    true_reviews = [
        review
        for review in original_reviews
        if review.get("unique_id") in true_ids
    ]

    with open(final_true_json_file, "w", encoding="utf-8") as f:
        json.dump(true_reviews, f, ensure_ascii=False, indent=2)

    print(f"Saved true use-case reviews: {len(true_reviews)}")
    print(f"Final JSON file: {final_true_json_file}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_prompt_file = args.output_dir / "first_prompt.txt"
    processed_csv_file = args.output_dir / "processed_use_case_predictions.csv"
    final_true_json_file = args.output_dir / "use_case_reviews_true_only.json"
    log_file = args.output_dir / "run_log.csv"

    original_reviews = load_json(args.input_file)
    prompt_template = load_prompt_template(args.prompt_file)
    few_shot_examples = load_json(args.few_shot_file)

    completed_ids = load_completed_unique_ids(processed_csv_file)

    remaining_reviews = [
        review
        for review in original_reviews
        if review.get("unique_id") not in completed_ids
    ]

    print(f"Input file: {args.input_file}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"Few-shot file: {args.few_shot_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total reviews in input file: {len(original_reviews)}")
    print(f"Already completed reviews: {len(completed_ids)}")
    print(f"Remaining reviews to process: {len(remaining_reviews)}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Few-shot examples used: {N_EXAMPLES}")
    print(f"Workers: {args.max_workers}")

    if not remaining_reviews:
        print("No remaining reviews. Creating final true-only JSON.")
        create_true_only_json(
            original_reviews=original_reviews,
            processed_csv_file=processed_csv_file,
            final_true_json_file=final_true_json_file,
        )
        return

    batches = list(chunk_list(remaining_reviews, BATCH_SIZE))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_batch = {}

        for i, (batch_number, batch_reviews) in enumerate(batches):
            save_first_prompt = i == 0

            future = executor.submit(
                process_batch,
                batch_number,
                batch_reviews,
                prompt_template,
                few_shot_examples,
                first_prompt_file,
                save_first_prompt,
            )

            future_to_batch[future] = batch_number

        for future in as_completed(future_to_batch):
            result = future.result()

            append_log_row(result["log_row"], log_file)

            if result["status"] == "success":
                append_rows_to_processed_csv(
                    result["processed_rows"],
                    processed_csv_file,
                )

                print(
                    f"Finished batch {result['batch_number']} | "
                    f"rows saved: {len(result['processed_rows'])}"
                )

            else:
                print(
                    f"Error in batch {result['batch_number']}: "
                    f"{result['log_row']['error_message']}"
                )

    create_true_only_json(
        original_reviews=original_reviews,
        processed_csv_file=processed_csv_file,
        final_true_json_file=final_true_json_file,
    )

    print("Done.")


if __name__ == "__main__":
    main()
