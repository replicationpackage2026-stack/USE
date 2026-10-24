
# Step 3: Task Consolidation

This directory contains the third step of the USE pipeline: **task consolidation**.

The usage scenarios extracted in the previous step contain open-ended task labels. Because semantically equivalent usages may be expressed using different task names, this step consolidates these labels into a consistent set of tasks within each usage domain.

The consolidation process consists of two stages:

1. **Task extraction** identifies a consolidated set of task labels for each domain.
2. **Task assignment** assigns the remaining/unmatched task labels to the consolidated task set.

## Directory Structure

```text
Step_3_task_consolidation/
├── prompts/
│   ├── extract.txt
│   └── assign.txt
├── assign_tasks.py
├── extract_tasks.py
└── readme.md
```

### Files

- `extract_tasks.py` — extracts and consolidates task labels within each domain.
- `assign_tasks.py` — assigns unmatched task labels to the consolidated task taxonomy.
- `prompts/extract.txt` — prompt used for task extraction/consolidation.
- `prompts/assign.txt` — prompt used for assigning unmatched tasks.

## 1. Extract Consolidated Tasks

The first stage processes task labels separately within each usage domain and extracts a consolidated set of tasks.

The following configuration was used for the study:

```bash
python3 extract_tasks.py \
  --domain_dir "<PATH_TO_DOMAIN_TASK_FILES>" \
  --domains \
    "Education & Research=Education_Research_tasks.csv" \
    "Multimedia=Multimedia_tasks.csv" \
    "Entertainment=Entertainment_tasks.csv" \
    "<NULL>=NULL_tasks.csv" \
    "Professional & Business=Professional_Business_tasks.csv" \
    "Personal & Social Support=Personal_Social_Support_tasks.csv" \
    "Writing=Writing_tasks.csv" \
    "Software Development=Software_Development_tasks.csv" \
    "Health & Wellness=Health_Wellness_tasks.csv" \
    "Language Translation=Language_Translation_tasks.csv" \
    "Legal=Legal_tasks.csv" \
  --prompt "prompts/extract.txt" \
  --results_dir "results/open_tasks_only" \
  --model "gpt-5.5"
```

### Main Arguments

- `--domain_dir`: Directory containing the task files separated by domain.
- `--domains`: Mapping between each domain name and its corresponding CSV file.
- `--prompt`: Prompt used to consolidate the open-ended task labels.
- `--results_dir`: Directory in which extraction results are stored.
- `--model`: OpenAI model used for task consolidation.

For the reported study, task extraction was performed using **GPT-5.5**.

## 2. Assign Remaining Tasks

After the consolidated task set is created, the assignment stage maps remaining/unmatched task labels to the consolidated tasks.

The following configuration was used for the study:

```bash
python3 assign_tasks.py \
  --mode full \
  --taxonomy_run_dir "results/open_tasks_only/run_005" \
  --domain_dir "<PATH_TO_UNASSIGNED_TASK_FILES>" \
  --prompt "prompts/assign.txt" \
  --results_dir "results/open_tasks_only/run_005/assignment_results" \
  --model "gpt-5.5" \
  --batch_size 100 \
  --workers 10 \
  --seed 42
```

### Main Arguments

- `--mode full`: Runs the complete assignment procedure.
- `--taxonomy_run_dir`: Directory containing the task taxonomy produced during the extraction stage.
- `--domain_dir`: Directory containing tasks that require assignment.
- `--prompt`: Prompt used to assign tasks to the consolidated taxonomy.
- `--results_dir`: Directory in which assignment results are stored.
- `--model`: OpenAI model used for task assignment.
- `--batch_size`: Number of task instances processed per batch.
- `--workers`: Number of parallel workers.
- `--seed`: Random seed used for reproducibility.

For the reported study, task assignment was performed using **GPT-5.5**, with a batch size of **100**, **10 workers**, and a random seed of **42**.

## Workflow

The expected execution order is:

```text
Domain-specific task labels
          │
          ▼
   extract_tasks.py
          │
          ▼
 Consolidated task set
          │
          ▼
    assign_tasks.py
          │
          ▼
Final consolidated task assignments
```

Run `extract_tasks.py` first. The resulting run directory is then supplied to `assign_tasks.py` through `--taxonomy_run_dir`.

## Prompts

The exact prompts used in the study are provided in the `prompts/` directory:

- `prompts/extract.txt`
- `prompts/assign.txt`

These files should be kept unchanged when reproducing the reported results, as changes to the prompts may affect the resulting task consolidation and assignment.
