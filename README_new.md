# Dirty ER Matching CLI - Usage Guide

## Overview

This CLI tool runs a comprehensive dirty Entity Resolution (ER) matching evaluation pipeline with support for multiple datasets, LLM models, and advanced configuration options. It builds a match graph, detects communities, and audits transitivity violations.

## Installation

### Prerequisites
- Python 3.11+
- pip or conda

### Setup

```bash
# Create and activate environment
conda env create -f environment.yaml
conda activate llm4em

# Or install requirements directly
pip install -r requirements.txt
```

## Basic Usage

### Command Structure

```bash
python -m src.dirty_matching.cli [OPTIONS]
```

### Required Arguments

- **`--output-dir`** (PATH) - **Required**. Directory to save all output files and results.

## Options Reference

### Dataset Configuration

#### `--dataset-name` (TEXT)
Choose the dataset for matching evaluation.
- **Choices**: `cora`, `wdc`, `cddb`, `musicbrainz`
- **Default**: `cora`
- **Description**: Select which benchmark dataset to run ER matching on.

```bash
# Run on WDC dataset
--dataset-name wdc

# Run on Cora (default)
--dataset-name cora
```

#### `--reader-root` (PATH)
Custom path to read dataset files from.
- **Default**: `None` (uses default dataset location)
- **Description**: Specify a custom root directory containing dataset files. Useful for using preprocessed or alternative data sources.

```bash
--reader-root /path/to/custom/data
```

#### `--candidates-csv` (PATH)
Path to a pre-computed candidates CSV file.
- **Default**: `None` (candidates generated automatically)
- **Description**: Provide pre-computed candidate pairs instead of generating them from scratch.

```bash
--candidates-csv /path/to/candidates.csv
```

### LLM Configuration

#### `--model-name` (TEXT)
Primary LLM model for matching evaluation.
- **Default**: `gpt-4o-mini`
- **Examples**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `claude-3-sonnet`
- **Description**: Specify the LLM model to use for entity matching decisions.

```bash
--model-name gpt-4o-mini
```

#### `--refine-model-name` (TEXT)
LLM model for the refine pipeline (community resolution and arbitration).
- **Default**: `gpt-5-mini`
- **Description**: Separate, typically more capable model used in stage2-4 of the refine pipeline.

```bash
--refine-model-name gpt-5-mini
```

### Retrieval Configuration

#### `--topk` (INTEGER)
Number of top candidates to retrieve for each record.
- **Default**: `20`
- **Description**: Controls the recall@k metric; higher values retrieve more candidates but increase LLM calls.

```bash
--topk 50  # Retrieve top 50 candidates instead of 20
```

#### `--force-rebuild-index`
Force rebuilding the SparseRetriever index before searching.
- **Default**: `False` (flag)
- **Description**: Use this flag to rebuild the retrieval index from scratch, useful if data has changed.

```bash
--force-rebuild-index
```

### Parallel Processing

#### `--max-workers` (INTEGER)
Maximum number of parallel workers for evaluation.
- **Default**: `16`
- **Description**: Controls concurrency when making LLM API calls. Higher values speed up processing but may hit rate limits.

```bash
--max-workers 32  # Use 32 parallel workers
```

### Sampling Configuration

#### `--sample-n` (INTEGER)
Number of records to sample for evaluation.
- **Default**: `200`
- **Description**: Randomly samples N records from the dataset. Use `--sample-frac` instead for percentage-based sampling.

```bash
--sample-n 500  # Sample 500 records
```

#### `--sample-frac` (FLOAT)
Fraction of dataset to sample (0.0-1.0).
- **Default**: `None` (uses `--sample-n` instead)
- **Description**: Sample a percentage of the dataset (e.g., 0.1 = 10%). Overrides `--sample-n` when specified.

```bash
--sample-frac 0.2  # Sample 20% of the dataset
```

#### `--sample-cluster-n` (INTEGER)
Sample N complete clusters (ground-truth communities).
- **Default**: `None`
- **Description**: Instead of sampling individual records, sample entire clusters to maintain ground-truth structure.

```bash
--sample-cluster-n 5  # Sample 5 complete clusters
```

#### `--sample-seed` (INTEGER)
Random seed for reproducible sampling.
- **Default**: `42`
- **Description**: Ensures reproducible results across runs.

```bash
--sample-seed 42
```

### Cluster Visualization

#### `--cluster-layout-k` (FLOAT)
Ideal spring-layout distance for ground-truth cluster centers.
- **Default**: `None` (auto-calculated)
- **Description**: Controls spatial distance between cluster centers in visualization.

```bash
--cluster-layout-k 0.5
```

#### `--cluster-layout-iterations` (INTEGER)
Spring-layout iterations for positioning cluster centers.
- **Default**: `350`
- **Description**: More iterations = better spatial arrangement but slower computation.

```bash
--cluster-layout-iterations 500
```

#### `--cluster-layout-spread` (FLOAT)
Global spread scale for cluster centers after normalization.
- **Default**: `None` (auto-calculated)
- **Description**: Controls overall visualization spread.

```bash
--cluster-layout-spread 1.5
```

#### `--cluster-layout-norm-quantile` (FLOAT)
Quantile for robust normalization of cluster-center distances.
- **Default**: `0.9`
- **Description**: Higher quantiles are more robust to outliers.

```bash
--cluster-layout-norm-quantile 0.95
```

### Violation Detection

#### `--violation-mode` (TEXT)
Strategy for detecting transitivity violations.
- **Choices**: `strict_negative`, `include_missing`
- **Default**: `strict_negative`
- **Description**:
  - `strict_negative`: Only records with explicit negative predictions count as violations
  - `include_missing`: Missing positive links also count as violations

```bash
--violation-mode strict_negative
```

### Refine Pipeline Options

#### `--enable-refine`
Enable the 4-stage refine ER pipeline.
- **Default**: `False` (flag)
- **Description**: Activates advanced community detection and resolution stages. Without this flag, only basic matching is performed.

```bash
--enable-refine
```

When enabled, the refine pipeline includes:
1. **Stage 1**: Community detection (Louvain/greedy/label propagation)
2. **Stage 2**: Local LLM resolution of small communities
3. **Stage 3**: Golden record synthesis
4. **Stage 4**: Global arbitration for macro conflicts

#### `--community-detection-method` (TEXT)
Algorithm for stage1 community partitioning.
- **Choices**: `louvain`, `greedy`, `label_prop`, `k_clique`, `spectral`
- **Default**: `louvain`
- **Description**: Different algorithms have different computational costs and quality trade-offs.

```bash
--community-detection-method louvain      # Best balance (default)
--community-detection-method greedy       # Faster but lower quality
--community-detection-method spectral     # Higher quality but slower
```

#### `--louvain-resolution` (FLOAT)
Resolution parameter for Louvain community detection.
- **Default**: `1.0`
- **Description**: Controls granularity of communities (lower = larger communities, higher = smaller communities).

```bash
--louvain-resolution 0.5   # Larger, coarser communities
--louvain-resolution 2.0   # Smaller, finer communities
```

#### `--louvain-seed` (INTEGER)
Random seed for Louvain algorithm.
- **Default**: `42`
- **Description**: Ensures reproducible community detection.

```bash
--louvain-seed 42
```

#### `--max-community-size` (INTEGER)
Maximum size threshold for communities before subdivision.
- **Default**: `10`
- **Description**: Communities larger than this are recursively subdivided.

```bash
--max-community-size 15
```

#### `--max-recursion-depth` (INTEGER)
Maximum depth for recursive community subdivision.
- **Default**: `6`
- **Description**: Prevents infinite recursion; stops after N subdivision levels.

```bash
--max-recursion-depth 8
```

#### `--resolution-scale` (FLOAT)
Recursive Louvain resolution multiplier (must be >1.0).
- **Default**: `1.25`
- **Description**: Multiplies resolution by this factor in each recursion level to find finer communities.

```bash
--resolution-scale 1.5  # More aggressive subdivision
```

### Stage 2: Local Resolution

#### `--local-resolve-max-size` (INTEGER)
Size threshold for sending communities to local LLM resolution.
- **Default**: `10`
- **Description**: Only communities with size ≤ this threshold are sent to LLM for resolution.

```bash
--local-resolve-max-size 15
```

#### `--local-resolve-workers` (INTEGER)
Parallel workers for local LLM resolution.
- **Default**: `8`
- **Description**: Number of concurrent LLM calls during stage2.

```bash
--local-resolve-workers 16
```

#### `--local-resolve-retries` (INTEGER)
Retry attempts for failed local resolutions.
- **Default**: `3`
- **Description**: Handles transient API failures.

```bash
--local-resolve-retries 5
```

### Stage 3: Golden Record

#### `--golden-method` (TEXT)
Strategy for golden record selection/synthesis.
- **Choices**: `medoid`, `llm`
- **Default**: `medoid`
- **Description**:
  - `medoid`: Select the most central record in each community
  - `llm`: Use LLM to synthesize a golden record

```bash
--golden-method medoid   # Faster, use existing record
--golden-method llm      # More comprehensive, synthesize new record
```

#### `--golden-workers` (INTEGER)
Parallel workers for golden record operations.
- **Default**: `8`
- **Description**: Number of concurrent workers during stage3.

```bash
--golden-workers 16
```

#### `--golden-retries` (INTEGER)
Retry attempts for golden record operations.
- **Default**: `3`
- **Description**: Handles transient API failures.

```bash
--golden-retries 5
```

### Stage 4: Global Arbitration

#### `--enable-global-arbitration` / `--disable-global-arbitration`
Enable/disable LLM arbitration for macro conflicts.
- **Default**: `--enable-global-arbitration` (enabled)
- **Description**: Stage4 uses LLM to resolve conflicts between different community resolution strategies.

```bash
--enable-global-arbitration    # Enable stage4 (default)
--disable-global-arbitration   # Skip stage4
```

#### `--global-resolve-workers` (INTEGER)
Parallel workers for global LLM arbitration.
- **Default**: `8`
- **Description**: Number of concurrent workers during stage4.

```bash
--global-resolve-workers 16
```

#### `--global-resolve-retries` (INTEGER)
Retry attempts for global arbitration.
- **Default**: `3`
- **Description**: Handles transient API failures.

```bash
--global-resolve-retries 5
```

#### `--max-global-conflicts` (INTEGER)
Maximum number of pair/triangle conflicts to arbitrate.
- **Default**: `200`
- **Description**: Limits conflicts sent to stage4 to control costs.

```bash
--max-global-conflicts 500
```

#### `--refine-max-stage` (INTEGER)
Run refine pipeline only up to this stage.
- **Range**: 1-4
- **Default**: `4`
- **Description**: Useful for debugging or running partial pipelines.

```bash
--refine-max-stage 2  # Run only stages 1-2
--refine-max-stage 3  # Run stages 1-3, skip stage 4
```

## Usage Examples

### Example 1: Basic Evaluation
Run a simple evaluation on the Cora dataset with default settings:

```bash
python -m src.dirty_matching.cli \
  --output-dir ./results/cora_basic
```

### Example 2: Custom Model and Dataset
Use GPT-4o model on WDC dataset with sampling:

```bash
python -m src.dirty_matching.cli \
  --dataset-name wdc \
  --model-name gpt-4o \
  --sample-n 500 \
  --output-dir ./results/wdc_gpt4o
```

### Example 3: Full Refine Pipeline with Community Detection
Run the complete 4-stage refine pipeline with Louvain community detection:

```bash
python -m src.dirty_matching.cli \
  --dataset-name cora \
  --model-name gpt-4o-mini \
  --enable-refine \
  --community-detection-method louvain \
  --louvain-resolution 1.0 \
  --max-community-size 10 \
  --golden-method llm \
  --enable-global-arbitration \
  --output-dir ./results/cora_refine_full \
  --max-workers 16
```

### Example 4: Aggressive Community Subdivision
Run with finer-grained community detection:

```bash
python -m src.dirty_matching.cli \
  --dataset-name cora \
  --enable-refine \
  --community-detection-method louvain \
  --louvain-resolution 2.0 \
  --resolution-scale 1.5 \
  --max-community-size 5 \
  --max-recursion-depth 8 \
  --output-dir ./results/cora_fine_grained
```

### Example 5: Budget-Conscious Evaluation
Minimal LLM calls with medoid golden records:

```bash
python -m src.dirty_matching.cli \
  --dataset-name cora \
  --model-name gpt-4o-mini \
  --sample-frac 0.1 \
  --topk 10 \
  --enable-refine \
  --local-resolve-max-size 5 \
  --golden-method medoid \
  --disable-global-arbitration \
  --refine-max-stage 2 \
  --output-dir ./results/cora_budget
```

### Example 6: High-Quality Evaluation
Maximum quality with best models and settings:

```bash
python -m src.dirty_matching.cli \
  --dataset-name cora \
  --model-name gpt-4o \
  --refine-model-name gpt-4o \
  --enable-refine \
  --community-detection-method spectral \
  --max-workers 32 \
  --local-resolve-workers 16 \
  --golden-method llm \
  --enable-global-arbitration \
  --global-resolve-workers 16 \
  --max-global-conflicts 500 \
  --output-dir ./results/cora_hq
```

### Example 7: Reproducible Run with Sampling
Run with complete reproducibility:

```bash
python -m src.dirty_matching.cli \
  --dataset-name cddb \
  --sample-cluster-n 3 \
  --sample-seed 42 \
  --louvain-seed 42 \
  --topk 20 \
  --enable-refine \
  --output-dir ./results/cddb_reproducible
```

## Output Files

After running the CLI, the `--output-dir` will contain:

```
output-dir/
├── matches.json              # Final matching decisions
├── match_graph.pkl           # Serialized match graph
├── communities.json          # Detected communities (if refine enabled)
├── transitivity_violations.json  # Detected violations
├── golden_records.json       # Golden records (if stage 3 completed)
├── conflicts.json            # Macro conflicts (if stage 4 completed)
├── evaluation_metrics.json   # Performance metrics
└── config.json              # Configuration used for this run
```

## Performance Tips

### Speed Optimization
- **Reduce `--topk`**: Fewer candidates = fewer LLM calls
- **Increase `--max-workers`**: More parallel workers (watch for rate limits)
- **Disable `--refine-stages`**: Skip advanced stages if not needed
- **Use `--sample-frac`**: Test with smaller samples first

### Quality Optimization
- **Increase `--topk`**: Better recall = more accurate matches
- **Use better models**: `gpt-4o` > `gpt-4o-mini`
- **Enable `--refine`**: Full pipeline often produces better results
- **Tune `--louvain-resolution`**: Find sweet spot for your data

### Cost Optimization
- **Use `gpt-4o-mini`**: Cheaper than `gpt-4o`
- **Set `--local-resolve-max-size` low**: Only resolve small communities
- **Disable `--global-arbitration`**: Skip expensive stage 4
- **Use `--golden-method medoid`**: Faster than LLM synthesis

## Troubleshooting

### "Missing required argument: --output-dir"
You must specify an output directory:
```bash
--output-dir ./results/my_run
```

### "Unknown dataset: xyz"
Choose from: `cora`, `wdc`, `cddb`, or `musicbrainz`
```bash
--dataset-name cora
```

### API Rate Limit Errors
Reduce parallel workers:
```bash
--max-workers 8
```

### Out of Memory
Reduce sampling size and parallel workers:
```bash
--sample-frac 0.05
--max-workers 4
```

### Slow Performance
Check `--topk` value (more = slower). Also ensure `--max-workers` matches your CPU/API capacity.

## API Keys

The CLI uses OpenAI by default. Set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

For other LLM providers, modify the configuration in `src/dirty_matching/llm/`.

## Advanced Configuration

For more fine-grained control, edit the pipeline configuration in:
- `src/dirty_matching/pipeline.py`
- `src/dirty_matching/core/`
