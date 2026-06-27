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
