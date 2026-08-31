# Research notes index

This folder holds research notes produced while working on tasks in this repository. It exists so that knowledge gathered during a task (library behavior, model details, design decisions, gotchas) is not lost after the task ends.

This `README.md` is the manager for the folder: every note must be listed in the table below with a short summary of what it covers and why it was written. Keep this table in sync whenever a note is added, renamed, or removed.

## Conventions

- One note per topic. Use a descriptive kebab-case filename, e.g. `flux-text-encoder-caching.md`.
- Start each note with a title, the date, and the task or question that prompted it.
- Link sources (file paths, library versions, doc URLs) so the note can be verified later.

## Notes

| Note | Purpose |
| --- | --- |
| [oplora-and-full-model-anti-forgetting.md](oplora-and-full-model-anti-forgetting.md) | Deep dive on OPLoRA (orthogonal-projection LoRA) and whether its anti-forgetting guarantee can extend to full-parameter fine-tuning; compares full-model alternatives (Rank-1 EWC, replay, distillation, gradient/null-space projection, OFT) for diffusion. |
| [oplora-implementation-plan.md](oplora-implementation-plan.md) | Detailed, codebase-grounded plan for adding OPLoRA (LoRA only) to diffusion-pipe: hook points, new `utils/oplora.py` projector, config keys, edge cases (quantization, block swap, pipeline sharding), and the CPU test plus GPU verification plan. Pre-implementation. |
| [upstream-api-drift-audit.md](upstream-api-drift-audit.md) | Why code that copies or subclasses upstream internals breaks when the upstream moves — covering both `submodules/` pins and pip dependencies (`torch`, `bitsandbytes`). Holds the dependency→file maps, the audit procedure to run after any submodule bump or dependency upgrade, and the two guard scripts (`tools/check_comfy_signatures.py`, `tools/check_vendored_apis.py`). Includes the 2026-06-29 ComfyUI and 2026-07-29 bitsandbytes/torch audit results. |
| [lumina2-vs-sd-scripts-training.md](lumina2-vs-sd-scripts-training.md) | Side-by-side comparison of the Lumina2 training implementation in `models/lumina_2.py` vs the Lumina2 trainer in the `sd-scripts` tree (`lumina_train_network.py` and friends): model loading, tokenization, timestep sampling/noise formula, forward pass, loss, and LoRA targeting. Same core math, different training harness (DeepSpeed pipeline parallelism vs `accelerate`). |
