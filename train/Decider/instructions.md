# ReDence TEv1 Decider training

From the repository root:

```bash
python train/Decider/thinktrain.py --validate_only
python train/Decider/thinktrain.py
```

The default model is `togethercomputer/Tev1-4B-experimental`, not a generic Qwen checkpoint. The loader reads the `examples` array in `tev1_training_data.json`, preserving its domain-group partitions: 4,269 train / 559 validation / 577 test examples.

Defaults follow the paper: 2,048 tokens, LoRA rank 8 / alpha 16, one epoch, learning rate `5e-5`, frozen NF4 backbone. Thinking is disabled. Only the final option letter (A=true, B=false) and EOS are supervised; training and controller share the same system instruction and serialized state/question/options. Recommended sampling weights affect training only. `--evaluate_test` explicitly evaluates test loss after training.

The adapter, tokenizer, and serving metadata are saved to `models/decider/`. See the root README for serving, controller use, and the missing browser/final-decision supervision in this release.
