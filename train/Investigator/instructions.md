# ReDence Investigator training

From the repository root:

```bash
python train/Investigator/thinktrain.py --validate_only
python train/Investigator/thinktrain.py
```

Defaults: `Qwen/Qwen3.5-4B`, bundled trajectory ZIP, 16,384 tokens, LoRA rank 16 / alpha 32, three epochs, learning rate `1e-4`, full context, and assistant-completion-only loss. Both models preserve the TEv1 bundle's domain-group train/validation/test assignments.

Preparation replaces exactly matching recorded source selections with `delegate_decision` requests, then returns the recorded selection and tool observation. It does not invent browser or final-verdict labels. Overlength examples are counted and excluded. Use `--context_mode short --tool_history_exchanges 2` for compact tool history, or `--help` for all options.

The trained adapter and tokenizer are saved to `models/investigator/`. Serve it alongside the trained TEv1 adapter with `python -m redence.serve_models`; see the root README for installation, inference, dataset limitations, and validation.
