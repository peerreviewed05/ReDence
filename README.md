# ReDence

ReDence investigates websites for social engineering scams and produces evidence-linked reports. It combines a **Qwen3.5-4B Investigator** for planning, interpretation, and reporting with a **TEv1-4B Decider** for bounded decisions. A Python **Controller** routes decisions, executes tools, records observations, and validates reports. The controller is orchestration code; the two trainable models are Investigator and Decider.

## Repository layout

- `redence/`: controller, adapter server, browser tools, investigative prompts, WHOIS utilities, and URL-list runner.
- `redence/prompt.txt`: Investigator prompt used directly by the controller, covering EXP, PLAN, ADAPT, REPORT, and Decider coordination.
- `train/Investigator/thinktrain.py`: Qwen3.5-4B training entry point.
- `train/Decider/thinktrain.py`: TEv1-4B training entry point; reads the bundled `tev1_training_data.json` directly.
- `train/Controller/controller.py`: convenient entry point for the controller.
- `train/data.py`, `train/sft.py`, `train/trajectory.py`: shared data preparation, completion-only QLoRA training, and context helpers.
- `data/training_data_reasoning.zip`: 1,539 teacher investigation trajectories.
- `data/url_dataset_main.csv`, `data/url_whois_mapping.json`, `data/whois/`: supplied URL metadata and WHOIS records. CSV category values are original annotations, not a clean binary benchmark split.
- `tests/`: dependency-free data/protocol/controller regression tests.

Training creates `models/investigator/` and `models/decider/`; checkpoints and investigation results go under `outputs/`. Trained weights are not included.

## Installation

Run commands from the repository root. Use Python 3.11 or 3.12. Training and default 4-bit serving require an NVIDIA CUDA GPU with enough memory for the selected models and context lengths. Install a PyTorch build appropriate for your CUDA environment first if necessary.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r train/requirements.txt -r redence/requirements.txt
python -m playwright install chromium
cp redence/env_sample redence/.env
```

The browser defaults to bundled Chromium. Set `PLAYWRIGHT_CHANNEL=chrome` to use installed Chrome. The vendored Playwright CAPTCHA tools require system `ffmpeg` for audio conversion; do not install the external `selenium-captcha-processing` package over this fork. Ollama is optional when using exported models instead of the included adapter server.

## Validate the supplied data

These checks need only Python's standard library and do not load model weights:

```bash
python train/Decider/thinktrain.py --validate_only
python train/Investigator/thinktrain.py --validate_only
python -m unittest discover -s tests -v
```

## Train both models

```bash
python train/Decider/thinktrain.py
python train/Investigator/thinktrain.py
```

The Decider starts from [Together AI's TEv1-4B checkpoint](https://huggingface.co/togethercomputer/Tev1-4B-experimental), uses a 2,048-token limit, LoRA rank 8 / alpha 16, one epoch, and learning rate `5e-5`. It learns the supplied **A = true / B = false** selections with thinking disabled. Only the answer and EOS token contribute to loss; provenance, split assignments, sampling weights, and auxiliary targets are excluded from model input. Recommended training-row sampling weights are applied by default; validation and test remain unresampled.

The Investigator starts from `Qwen/Qwen3.5-4B`, uses a 16,384-token limit, LoRA rank 16 / alpha 32, three epochs, and learning rate `1e-4`. It trains on one assistant completion per trajectory turn with all context and tool observations masked from loss. Preparation replaces **1,684** exactly reconstructable search-source selections with controller delegation requests and wraps their recorded observations with the returned decision. Another **22** candidates have differing recorded tool arguments and retain their original actions.

Both models use frozen NF4 4-bit backbones and PEFT adapters. Overlength examples are counted and excluded; questions, choices, and answer targets are never silently cut. Saved adapters include tokenizers and `redence_metadata.json`, which the model server uses to recover the base model, role, thinking setting, and sequence limit.

Useful options, available on both entry points:

```bash
python train/Decider/thinktrain.py --help
python train/Investigator/thinktrain.py --help
python train/Investigator/thinktrain.py --context_mode short --tool_history_exchanges 2
python train/Decider/thinktrain.py --evaluate_test
```

Use `--data_file`, `--model_name`, `--output_dir`, and `--lora_dir` to override paths and checkpoints. Defaults resolve relative to the repository, so training entry points also work from other working directories. `--max_examples` caps training rows only for debugging. `--evaluate_test` explicitly evaluates held-out test loss after training; it does not compute website detection accuracy. Weighted sampling currently supports one GPU; disable it with `--no-weighted_sampling` when using distributed training.

## Serve the trained adapters

Start the server after training. It loads each adapter onto its own recorded base checkpoint, retaining Investigator image support.

```bash
python -m redence.serve_models \
  --investigator-adapter models/investigator \
  --decider-adapter models/decider
```

The local endpoint is `http://127.0.0.1:11435/api/chat`, with model names `investigator` and `decider`. Model loading happens before investigations begin. The server applies the saved thinking settings and rejects inputs that exceed each model's context limit. Keep delegated state concise; no automatic removal of evidence or choices is performed.

Alternatively, serve correctly exported base-plus-adapter models through Ollama, then configure `REDENCE_MODEL_HOST`, `REDENCE_INVESTIGATOR_MODEL`, and `REDENCE_DECIDER_MODEL` in `redence/.env` or pass the controller flags below. A PEFT adapter directory alone is not an Ollama model.

## Investigate a website

In another terminal with the environment activated:

```bash
python -m redence.controller \
  --url https://example.org \
  --output-dir outputs/example
```

The equivalent convenience entry point is:

```bash
python train/Controller/controller.py --url https://example.org --output-dir outputs/example-other
```

Additional options:

- `--budget 120`: per-website wall-clock budget, including browser initialization and model requests. A supervisor stops the worker process group on expiry and writes an incomplete report; process cleanup may take a further second.
- `--domain-intel path/to/whois_intel.txt`: inject previously collected WHOIS/DNS/SSL evidence.
- `--archive-url http://localhost:8080/replay/...`: HTTP(S) archive replay to try when the live website cannot be retrieved. WACZ replay must be served separately.
- `--categories-file path/to/categories.json`: JSON array of category names. Supply the paper's BBB catalog for that taxonomy; the default is the eight broad categories in the supplied investigative prompt. Catalogs exceeding 25 categories use a group-selection step before category selection.
- `--investigator-model NAME --decider-model NAME --model-host URL`: override the configured model server and model aliases.

For a URL list (one URL per line, blank lines and `#` comments accepted):

```bash
python redence/run_urls_from_txt.py urls.txt --limit 10 --output-dir outputs/batch
```

