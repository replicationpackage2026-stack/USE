import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "gpt-5.5"

# Update if pricing changes.
PRICING = {
    "gpt-5.5": {
        "input": 5.00 / 1_000_000,
        "cached_input": 0.50 / 1_000_000,
        "output": 30.00 / 1_000_000,
    },
    "gpt-5-mini": {
        "input": 0.25 / 1_000_000,
        "cached_input": 0.025 / 1_000_000,
        "output": 2.00 / 1_000_000,
    },
    "gpt-5-nano": {
        "input": 0.05 / 1_000_000,
        "cached_input": 0.005 / 1_000_000,
        "output": 0.40 / 1_000_000,
    },
}


# ============================================================
# Fixed output schema
# ============================================================

TASK_TAXONOMY_SCHEMA = {
    "type": "json_schema",
    "name": "task_taxonomy",
    "strict": True,
    "schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string"
                            },
                            "description": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "task",
                            "description"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["tasks"],
            "additionalProperties": False
        }
}


# ============================================================
# Input preparation
# ============================================================

def read_tasks(csv_path):
    """
    Read Task labels from one Domain CSV.

    Required columns:
        task
        frequency
    """

    df = pd.read_csv(csv_path)

    required_columns = {"task", "frequency"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: "
            f"{sorted(missing)}"
        )

    # Remove null / empty Tasks.
    df = df[df["task"].notna()].copy()

    df["task"] = (
        df["task"]
        .astype(str)
        .str.strip()
    )

    df = df[df["task"] != ""]

    return df


def format_task_input(df, include_frequency=False):
    """
    Format Tasks compactly, with one Task per line.
    """

    if include_frequency:
        lines = ["task; frequency"]

        for _, row in df.iterrows():
            task = row["task"]
            frequency = int(row["frequency"])

            lines.append(f"{task}; {frequency}")

    else:
        lines = []

        for task in df["task"]:
            lines.append(task)

    return "\n".join(lines)

# ============================================================
# Prompt construction
# ============================================================

def load_prompt_beginning(prompt_path):
    """
    The prompt file contains ONLY the experimental portion
    of the prompt.

    Everything after it is fixed by this script.
    """

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(
    prompt_beginning,
    domain,
    formatted_tasks,
    include_frequency=False
):
    """
    Construct the complete prompt.

    Structure:

        [EXPERIMENTAL PROMPT TEXT]

        [FIXED INPUT EXPLANATION]

        [FIXED OUTPUT INSTRUCTIONS]

        [TASK INPUT]
    """

    # --------------------------------------------------------
    # Fixed explanation of the input
    # --------------------------------------------------------

    if include_frequency:
        input_explanation = f"""
The Tasks above are from the {domain} Domain. Each Task is provided on a separate line in the format "task; frequency". Frequency is the number of times the Task occurs in the data. Higher-frequency Tasks represent more commonly reported uses and should receive greater consideration when determining which Tasks are important to represent.
""".strip()


    else:

        input_explanation = f"""
The Task labels below are from the {domain} Domain.

Each line represents one Task label. Task labels are separated by new lines.
""".strip()

    # --------------------------------------------------------
    # Fixed return instructions
    # --------------------------------------------------------

    output_instructions = """
Return the result using the following structure:

{
  "tasks": [
    {
      "task": "Task Name",
      "description": "Brief description of the Task."
    }
  ]
}

The "tasks" array must contain the Task taxonomy requested above.
Each Task must have a concise Task name and a brief description.
""".strip()

    # --------------------------------------------------------
    # Full prompt
    # --------------------------------------------------------

    full_prompt = f"""
{prompt_beginning}

{input_explanation}

{output_instructions}

TASK LABELS:

{formatted_tasks}
""".strip()

    return full_prompt


# ============================================================
# Domain specification
# ============================================================

def parse_domain_spec(spec):
    """
    Expected format:

        DOMAIN=FILENAME

    Example:

        Education & Research=Education_Research_tasks.csv
    """

    if "=" not in spec:
        raise ValueError(
            f"Invalid domain specification:\n"
            f"  {spec}\n\n"
            f"Expected:\n"
            f'  "Domain Name=filename.csv"'
        )

    domain, filename = spec.split("=", 1)

    domain = domain.strip()
    filename = filename.strip()

    if not domain:
        raise ValueError(
            f"Domain name is empty in: {spec}"
        )

    if not filename:
        raise ValueError(
            f"Filename is empty in: {spec}"
        )

    return domain, filename


# ============================================================
# Run directories
# ============================================================

def get_next_run_dir(base_results_dir):
    """
    Automatically create:

        run_001
        run_002
        run_003
        ...
    """

    os.makedirs(base_results_dir, exist_ok=True)

    existing = []

    for name in os.listdir(base_results_dir):

        if not name.startswith("run_"):
            continue

        try:
            number = int(name.split("_")[1])
            existing.append(number)
        except (IndexError, ValueError):
            continue

    next_number = max(existing, default=0) + 1

    run_dir = os.path.join(
        base_results_dir,
        f"run_{next_number:03d}"
    )

    os.makedirs(run_dir)

    return run_dir


# ============================================================
# Responses API helpers
# ============================================================

def extract_response_text(response):
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
    usage = getattr(response, "usage", None)

    if usage is None:
        return 0, 0, 0

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    cached_input_tokens = 0
    input_details = getattr(usage, "input_tokens_details", None)

    if input_details is not None:
        cached_input_tokens = getattr(
            input_details,
            "cached_tokens",
            0
        ) or 0

    return input_tokens, cached_input_tokens, output_tokens


# ============================================================
# Cost calculation
# ============================================================

def calculate_cost(
    model,
    input_tokens,
    cached_input_tokens,
    output_tokens
):
    if model not in PRICING:
        return None

    prices = PRICING[model]

    non_cached_input = max(
        input_tokens - cached_input_tokens,
        0
    )

    return (
        non_cached_input * prices["input"]
        + cached_input_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    )


# ============================================================
# API call
# ============================================================

def extract_taxonomy(
    client,
    model,
    full_prompt
):
    """
    GPT-5-family call using the Responses API.

    Temperature and reasoning effort are intentionally omitted
    so the model uses the same default configuration as the
    earlier Step 1 GPT-5 model comparison.
    """

    response = client.responses.create(
        model=model,
        input=full_prompt,
        text={
            "format": TASK_TAXONOMY_SCHEMA
        },
    )

    content = extract_response_text(response)

    if not content:
        raise ValueError("The model returned no text output.")

    parsed = json.loads(content)

    return response, parsed


# ============================================================
# Saving
# ============================================================

def save_text(path, text):

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path, obj):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract a Task taxonomy from "
            "Domain-level Task CSV files."
        )
    )

    parser.add_argument(
        "--domain_dir",
        required=True,
        help=(
            "Directory containing the Domain Task CSV files."
        )
    )

    parser.add_argument(
        "--domains",
        nargs="+",
        required=True,
        help=(
            'Domain/file pairs in the form '
            '"Domain Name=filename.csv"'
        )
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help=(
            "TXT file containing the experimental "
            "beginning of the prompt."
        )
    )

    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory in which runs are stored."
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model. Default: {DEFAULT_MODEL}"
    )

    parser.add_argument(
        "--include_frequency",
        action="store_true",
        help=(
            "Include each Task's frequency in the input. "
            "If omitted, only Task labels are provided."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    client = OpenAI()

    prompt_beginning = load_prompt_beginning(
        args.prompt
    )

    run_dir = get_next_run_dir(
        args.results_dir
    )

    print("=" * 70)
    print("TASK TAXONOMY EXTRACTION")
    print("=" * 70)

    print(f"Model:             {args.model}")
    print(f"Prompt:            {args.prompt}")
    print(f"Domain directory:  {args.domain_dir}")
    print(f"Frequency included:{args.include_frequency}")
    print(f"Run directory:     {run_dir}")
    print(f"Domains requested: {len(args.domains)}")

    # Save experimental prompt once.
    save_text(
        os.path.join(
            run_dir,
            "prompt_beginning.txt"
        ),
        prompt_beginning
    )

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "prompt_file": str(args.prompt),
        "domain_dir": str(args.domain_dir),
        "include_frequency": args.include_frequency,
        "domains": []
    }

    # --------------------------------------------------------
    # Process Domains
    # --------------------------------------------------------

    for spec in args.domains:

        domain, filename = parse_domain_spec(spec)

        csv_path = os.path.join(
            args.domain_dir,
            filename
        )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Input file not found: {csv_path}"
            )

        print("\n" + "=" * 70)
        print(f"DOMAIN: {domain}")
        print("=" * 70)

        df = read_tasks(csv_path)

        print(f"File:        {filename}")
        print(f"Task labels: {len(df):,}")

        formatted_tasks = format_task_input(
            df,
            include_frequency=args.include_frequency
        )

        full_prompt = build_prompt(
            prompt_beginning=prompt_beginning,
            domain=domain,
            formatted_tasks=formatted_tasks,
            include_frequency=args.include_frequency
        )

        # Safe directory name.
        safe_domain = (
            filename
            .replace("_tasks.csv", "")
            .replace(".csv", "")
        )

        domain_output_dir = os.path.join(
            run_dir,
            safe_domain
        )

        os.makedirs(
            domain_output_dir,
            exist_ok=True
        )

        # Save exact input sent to the model.
        save_text(
            os.path.join(
                domain_output_dir,
                "full_prompt.txt"
            ),
            full_prompt
        )

        save_text(
            os.path.join(
                domain_output_dir,
                "task_input.txt"
            ),
            formatted_tasks
        )

        print("Sending request...")

        response, taxonomy = extract_taxonomy(
            client=client,
            model=args.model,
            full_prompt=full_prompt
        )

        # ----------------------------------------------------
        # Usage / cost
        # ----------------------------------------------------

        (
            input_tokens,
            cached_input_tokens,
            output_tokens
        ) = extract_usage(response)

        cost = calculate_cost(
            model=args.model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens
        )

        number_extracted = len(
            taxonomy["tasks"]
        )

        print(
            f"Tasks returned: "
            f"{number_extracted:,}"
        )

        print(
            f"Input tokens:  "
            f"{input_tokens:,}"
        )

        print(
            f"Output tokens: "
            f"{output_tokens:,}"
        )

        if cost is not None:
            print(
                f"Estimated cost: ${cost:.6f}"
            )

        # ----------------------------------------------------
        # Save taxonomy
        # ----------------------------------------------------

        save_json(
            os.path.join(
                domain_output_dir,
                "taxonomy.json"
            ),
            taxonomy
        )

        # Convenient CSV version.
        taxonomy_df = pd.DataFrame(
            taxonomy["tasks"]
        )

        taxonomy_df.to_csv(
            os.path.join(
                domain_output_dir,
                "taxonomy.csv"
            ),
            index=False
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "domain": domain,
            "input_file": filename,
            "input_path": csv_path,
            "number_of_input_tasks": len(df),
            "include_frequency": args.include_frequency,
            "model": args.model,
            "api": "responses",
            "temperature": None,
            "reasoning_effort": "default",
            "number_of_unique_tasks": number_extracted,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": cost,
        }

        save_json(
            os.path.join(
                domain_output_dir,
                "metadata.json"
            ),
            metadata
        )

        run_summary["domains"].append(
            metadata
        )

    # --------------------------------------------------------
    # Run summary
    # --------------------------------------------------------

    save_json(
        os.path.join(
            run_dir,
            "run_summary.json"
        ),
        run_summary
    )

    print("\n" + "=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
