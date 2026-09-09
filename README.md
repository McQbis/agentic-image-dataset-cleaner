# agentic-image-dataset-cleaner
Remove low-quality photos from your dataset faster with the help of AI agents.

```bash
python -m pytest
```

## Key Findings and Design Decisions

During validation, several issues were identified and addressed:

* **Exposure-independent sharpness:** `blur_variance` is computed on a histogram-equalized image to prevent underexposure from being misclassified as blur. The raw Laplacian variance is preserved as `blur_variance_raw` for diagnostics.
* **Robust scoring:** `composite_score` uses a max-floor in addition to the weighted sum, preventing a catastrophic defect on one metric from being cancelled out by artificially good values on correlated metrics. `weighted_sum_score` is used as a deterministic tiebreaker.
* **Deterministic deduplication:** Near-duplicates are removed before scoring and VLM evaluation. Connected pHash components are clustered and the sharpest representative is kept, preventing duplicates from consuming the VLM budget.
* **Anti-anchoring:** Agent 3 is explicitly instructed not to equate worst-N membership with deletion. The application also detects suspiciously uniform recommendations and emits a warning.
* **Metric directions are deterministic:** Agent 1 no longer chooses whether a metric is `low_is_bad` or `high_is_bad`. Directions are defined in code from the mathematical meaning of each metric, while the agent only selects relevant metrics and their weights.
* **Tail-aware criteria selection:** Agent 1 receives explicit severity hints showing the relevant bad-tail percentile and its ratio to the median. This prevents metrics such as `blur_variance` from being incorrectly dismissed based only on their median.
* **Clean validation dataset:** Standalone degradations are generated from images without sharp counterparts in the dataset, separating scoring validation from deduplication validation. On the clean benchmark, scoring identified **14/15 standalone degradations in worst-20**, while the corrected metric direction brought **5/5 severely blurred samples** into worst-20.
