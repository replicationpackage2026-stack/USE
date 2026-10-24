# Step 2: Extract Usage Scenarios

This step extracts structured usage scenarios from Gen-AI app reviews identified as discussing usage in Step 1.

A review may contain zero, one, or multiple usage scenarios. Each extracted usage scenario is characterized by:

* **Domain:** the broader context or area of use.
* **Task:** the general user goal or type of work.
* **Activity:** the specific action through which the user accomplishes the task.
* **Use Status:** whether the usage is actual or desired.

Task and Activity are required for each extracted usage scenario. Domain and Use Status may be `null` when they cannot be determined from the review.

## Files

* `extract_usage_scenario.py`
  Main script for extracting structured usage scenarios from reviews using the OpenAI API. The script processes reviews in batches, validates model responses, checkpoints completed reviews, retries unresolved reviews, and records processing statistics.

* `prompt.txt`
  Prompt used to define the usage-scenario extraction task and the Domain, Task, Activity, and Use Status attributes.

## Input

The script takes a JSON file containing a list of review objects. Each review must contain:

* `reviewId`: unique identifier for the review.
* `content`: review text.

Other review metadata may also be present. Only the review ID and review content are sent to the model; the original metadata is retained in the output.

## Output

Results are stored in a run directory such as:

```text
results/full/run_001/
```

The main output is:

* `reviews_with_usage_dimensions.jsonl`
  Final dataset containing the original review information together with the extracted `usages` for each review.

Each usage has the following structure:

```json
{
  "domain": "Education & Research",
  "task": "Study Assistance",
  "activity": "Getting help with studies",
  "use_status": "actual"
}
```

A review for which no valid usage scenario is identified is represented with:

```json
"usages": []
```

The run directory also contains processing files used for reproducibility, logging, and recovery, including the saved prompt and run configuration, checkpointed results, request and retry logs, response anomalies, token and cost information, and a final run summary.

## Running the Full Dataset

Set the OpenAI API key before running the script:

```bash
export OPENAI_API_KEY="your-api-key"
```

Run the extraction on the full input dataset with:

```bash
python extract_usage_scenario.py \
  --mode full \
  --input_file path/to/usage_scenario_reviews.json \
  --prompt_file prompt.txt \
  --output_dir results
```

The model, batch size, number of concurrent workers, and retry settings are configured in `extract_usage_scenario.py`.

## Resuming an Interrupted Run

Completed reviews are saved incrementally to a checkpoint file. If processing is interrupted, the existing run can be resumed without reprocessing reviews that have already been successfully completed.

For example:

```bash
python extract_usage_scenario.py \
  --mode full \
  --input_file path/to/usage_scenario_reviews.json \
  --prompt_file prompt.txt \
  --output_dir results \
  --resume_run results/full/run_001
```

The resumed run uses the existing checkpoint and processing logs to identify completed and unresolved reviews. Reviews without a valid result are retried up to the configured maximum number of processing attempts.

The final `reviews_with_usage_dimensions.jsonl` file is created only when all input reviews have been successfully processed.
