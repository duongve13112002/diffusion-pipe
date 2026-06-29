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
| [comfyui-submodule-signature-audit.md](comfyui-submodule-signature-audit.md) | Why ComfyUI-backed models break when the `submodules/ComfyUI` pin moves (they re-implement forward and call ComfyUI leaves directly), the exact step-by-step audit to run after every submodule update, and the automated guard `tools/check_comfy_signatures.py`. Includes the 2026-06-29 audit results. |
