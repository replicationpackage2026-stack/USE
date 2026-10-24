# Step 1: Filter Usage Scenario Reviews

This step identifies Gen-AI app reviews that describe **usage scenarios**, i.e., reviews that provide information about how a user employs a Gen-AI application.

The filtering approach uses a Large Language Model (LLM) with a few-shot prompt to classify each review as either describing a usage scenario or not.

## Folder Structure

```text
Step_1_filter_reviews/
│
├── data/
│   └── [evaluation samples]
│
├── few_shot_examples.txt
├── filter_usage_scenario_reviews.py
├── prompt.txt
└── readme.md
```

## Files

- `filter_usage_scenario_reviews.py`: main script for identifying reviews that describe usage scenarios.
- `prompt.txt`: prompt used to classify reviews.
- `few_shot_examples.txt`: few-shot examples provided to the LLM.
- `data/`: manually annotated review samples used to evaluate the filtering approach.

## Evaluation Data

The `data/` directory contains the manually annotated samples used to evaluate the Step 1 filtering approach reported in the paper.

Each evaluation sample contains **96 reviews** with manually assigned ground-truth labels indicating whether each review describes a usage scenario.

The evaluation samples were used during the iterative development and evaluation of the filtering prompt. The final prompt and few-shot examples used by the pipeline are provided in this directory.

## LLM Filtering

The filtering script processes Gen-AI app reviews and determines whether each review contains evidence of a usage scenario.

### Inputs

The filtering pipeline requires:

- an input file containing the reviews to be classified;
- the filtering prompt in `prompt.txt`;
- the few-shot examples in `few_shot_examples.txt`; and
- an output directory for the resulting classifications.

### Usage

```bash
python filter_usage_scenario_reviews.py \
  --input_file <PATH_TO_INPUT_REVIEWS> \
  --prompt_file prompt.txt \
  --few_shot_file few_shot_examples.txt \
  --output_dir <PATH_TO_OUTPUT_DIRECTORY>
```

Additional runtime parameters can be configured through the script as needed.

## Output

The filtering process produces review-level classifications indicating whether each review describes a usage scenario. Reviews identified as containing usage scenarios are retained for the subsequent usage-scenario extraction stage of the USE pipeline.

## Reproducibility

The prompt, few-shot examples, filtering implementation, and manually annotated evaluation samples provided in this directory correspond to the Step 1 filtering procedure used for the experiments reported in the paper.
