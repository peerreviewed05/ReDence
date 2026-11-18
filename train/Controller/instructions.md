# ReDence Controller

Controller is Python orchestration code, not a third trainable model. Train Investigator and TEv1 Decider, start `python -m redence.serve_models` with their saved adapters, then run:

```bash
python train/Controller/controller.py --url https://example.org --output-dir outputs/example
```

Controller validates bounded requests, obtains an option label from Decider, executes the selected tool operation once, and returns the selection and resulting observation to Investigator. It keeps an observation trace and evidence ledger, delegates final assessment, validates JSON reports and evidence references, and bounds each website investigation to 120 seconds by default.

Use `--help` and the root README for configuration, archive replay, taxonomy selection, batch execution, model serving, and release limitations.
