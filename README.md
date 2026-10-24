# USE: Usage Scenario Extraction from Gen-AI App Reviews

This repository contains the replication package for **USE**, a framework for extracting and consolidating usage scenarios from user reviews of Generative AI (Gen-AI) mobile applications.

The package contains the prompts, scripts, and evaluation data used to implement and evaluate the USE pipeline reported in our MSR paper.


## Overview

USE transforms unstructured Gen-AI app reviews into structured usage scenarios through three main steps:

1. **Filter Usage-Scenario Reviews**  
   Identify reviews that contain information about how users employ Gen-AI applications.

2. **Extract Usage Scenarios**  
   Extract structured usage information from the reviews identified in Step 1.

3. **Consolidate Tasks**  
   Consolidate open-ended task labels into a consistent set of tasks and assign remaining task labels to the consolidated task set.


## Repository Structure

```text
USE-MSR2027/
│
├── Step_1_filter_reviews/
│   ├── data/
│   ├── few_shot_examples.txt
│   ├── filter_usage_scenario_reviews.py
│   ├── prompt.txt
│   └── readme.md
│
├── Step_2_extract_usage_scenario/
│   ├── extract_usage_scenario.py
│   ├── prompt.txt
│   └── readme.md
│
├── Step_3_task_consolidation/
│   ├── prompts/
│   │   ├── assign.txt
│   │   └── extract.txt
│   ├── assign_tasks.py
│   ├── extract_tasks.py
│   └── readme.md
│
└── README.md
```

Each step contains its own README with additional information about the corresponding scripts, inputs, outputs, prompts, and execution procedure.

## Step 1: Filter Usage-Scenario Reviews

`Step_1_filter_reviews/`

The first step identifies reviews that describe how users employ Gen-AI applications.

The filtering approach uses an LLM with a few-shot prompt to classify reviews according to whether they contain evidence of a usage scenario.

This directory contains:

- the filtering script;
- the prompt used for classification;
- the few-shot examples supplied to the model; and
- manually annotated evaluation samples used to evaluate the filtering approach.

See `Step_1_filter_reviews/readme.md` for details.

## Step 2: Extract Usage Scenarios

`Step_2_extract_usage_scenario/`

The second step transforms reviews identified in Step 1 into structured usage scenarios.

The LLM processes the review text using the provided extraction prompt and extracts the usage information required by the USE framework. A single review may contain more than one usage scenario.

This directory contains:

- the usage-scenario extraction script; and
- the prompt used for structured extraction.

See `Step_2_extract_usage_scenario/readme.md` for details.

## Step 3: Task Consolidation

`Step_3_task_consolidation/`

The third step consolidates the open-ended task labels produced during usage-scenario extraction.

This process consists of two stages:

1. extracting a consolidated set of task labels within each usage domain; and
2. assigning remaining/unmatched task labels to the consolidated task set.

This directory contains:

- the task extraction script;
- the task assignment script; and
- the prompts used for both stages.

See `Step_3_task_consolidation/readme.md` for details.

## Evaluation Data

The manually annotated data included in this repository are provided to support reproduction of the evaluations reported in the paper.

Step 1 evaluation data are located in:

```text
Step_1_filter_reviews/data/
```

Additional details about the evaluation samples and their format are provided in the Step 1 README.


## Running the Pipeline

The steps should be executed sequentially:

```text
Step 1 → Step 2 → Step 3
```

Each directory contains a dedicated README describing the required inputs, configuration, and execution command for that stage.

For reproduction, we recommend following the instructions in the individual directories in order:

```text
Step_1_filter_reviews/readme.md
        ↓
Step_2_extract_usage_scenario/readme.md
        ↓
Step_3_task_consolidation/readme.md
```

## Reproducibility

The scripts and prompts in this repository correspond to the USE configuration used for the experiments reported in the paper.

Because USE relies on hosted Large Language Models, exact outputs may vary due to model updates, API changes, and other sources of nondeterminism. The prompts and model configurations used in the reported experiments are provided to facilitate reproduction as closely as possible.

## Data and Privacy

The data included in this replication package consist of publicly available user reviews used for evaluation of the USE pipeline. The package does not include API credentials or other authentication information.
