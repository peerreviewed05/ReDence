# Running ReDence

Install `redence/requirements.txt` and `train/requirements.txt` in a Python 3.11/3.12 environment, then run `python -m playwright install chromium`. Copy `redence/env_sample` to `redence/.env` for local configuration.

From the repository root, train and serve both model adapters as described in the root README. Run one investigation with:

```bash
python -m redence.controller --url https://example.org --output-dir outputs/example
```

Or run a URL list with:

```bash
python redence/run_urls_from_txt.py urls.txt --limit 10 --output-dir outputs/batch
```

The controller writes an evidence-linked JSON report, an observation/decision trace, and screenshots. It enforces a 120-second default budget and returns `Insufficient evidence` on incomplete investigations. Use `--help` or the root README for model aliases, custom categories, domain intelligence, and archive replay.
