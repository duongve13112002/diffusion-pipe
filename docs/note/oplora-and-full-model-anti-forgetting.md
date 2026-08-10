# OPLoRA and full-model anti-forgetting

Date: 2026-06-27
Prompt: Research OPLoRA in depth and decide whether it can be applied to full-model (full-parameter) fine-tuning, or whether a better method exists. Cross-referenced with the local `anti-forgetting.md`.

## What OPLoRA actually is

Source: "OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting during Parameter-Efficient Fine-Tuning", arXiv 2510.13003 (v2), accepted to AAAI 2026.

OPLoRA constrains a LoRA update so it never touches the base weight's dominant singular directions. For a frozen weight `W0 = U S V^T`, it keeps the top-k left/right singular vectors `U_k`, `V_k` and projects the adapter:

```
ΔW = P_L (B A) P_R,   P_L = I - U_k U_k^T,   P_R = I - V_k V_k^T
```

Proposition 2 in the paper proves this preserves the top-k singular triples exactly: `U_k^T W' V_k = S_k`, i.e. `W' v_i = σ_i u_i` and `W'^T u_i = σ_i v_i` for `i ≤ k`. The metric `ρ_k = ||Q_k ΔW||_F^2 / ||ΔW||_F^2` quantifies leftover interference (reported `ρ_128 = 0.003` with randomized SVD, so not perfectly zero in practice).

Implementation cost: a one-time truncated SVD of each target weight at startup (~5.5 min on LLaMA-2 7B) and a few small matmuls per step. Wall clock on LLaMA-2 7B: LoRA 5h13m vs OPLoRA 6h12m. In the repo's `anti-forgetting.md` the same idea is realised as a post-optimizer correction of the small factors:

```
up'   = up   - U_k (U_k^T up)
down' = down - (down V_k) V_k^T
```

The paper explicitly leaves it as LoRA-only and gives **no** discussion of full-parameter fine-tuning.

## Does the math generalize to full fine-tuning?

Yes, the preservation theorem does not depend on the update being low-rank. If you apply the same double-sided projection to a full-rank update `ΔW` (the actual weight delta), then because `U_k^T P_L = 0` and `P_R V_k = 0`:

```
U_k^T ΔW' = U_k^T P_L ΔW P_R = 0
ΔW' V_k   = P_L ΔW P_R V_k   = 0
```

so `W' = W0 + ΔW'` keeps the top-k singular triples exactly, identical to the LoRA case. The guarantee carries over. What does **not** carry over for free is the cost and the maintenance of that guarantee.

### Why it is not "free" at full FT
1. Per-step compute. LoRA only has to correct two skinny factors `B (m×r)` and `A (r×n)`, costing roughly `O(k r (m+n))` per layer. A full update is a dense `m×n` matrix, so `P_L ΔW P_R` costs `O(k m n)` per layer, applied to every weight every step. On a 12B diffusion transformer that is a large recurring cost, not a startup one.
2. Memory. You must keep `U_k (m×k)` and `V_k (n×k)` resident for every projected weight. Per layer it is small, but it scales with layer count × k. Still well below a resident teacher copy, but it is real (the repo notes the same caveat for OPLoRA at LoRA scale).
3. Maintaining the guarantee under an adaptive optimizer. The clean proof assumes the projected delta is what actually lands on the weight. With Adam, per-coordinate second-moment scaling is applied after the gradient, so projecting the gradient does not keep the realised update inside the orthogonal complement (the same projection-vs-preconditioning tension that the Muon / AdaMuon line of work addresses). You must project the post-optimizer update, and because randomized SVD leakage and fp accumulation drift the top-k over many steps, you also need periodic re-orthogonalization of `W` against the snapshot. OPLoRA dodges all of this by re-deriving `up'`, `down'` from the small factors each step.

Conclusion: porting OPLoRA to full FT means building a "weight-space SVD orthogonal projection" method (project the post-optimizer full-weight update double-sidedly + re-orthogonalize periodically). Implementable, but it is effectively a heavier cousin of the gradient/null-space projection family, not a free flag flip.

## Full-model methods that already do "orthogonal projection"

The orthogonal-projection idea exists for full networks, but in gradient/feature space rather than on low-rank factors:

- Gradient Projection Memory (GPM), Saha et al., ICLR 2021: project gradients orthogonal to the subspace spanned by important input/activation directions of old tasks.
- Adam-NSCL, Wang et al., CVPR 2021: constrain updates to the null space of previous-task input feature covariance.
- GNSP, arXiv 2507.19839 (2025): gradient null space projection, applied to VLMs to preserve cross-modal alignment.
- ROGO, restricted orthogonal gradient projection (relaxes the hard constraint to allow forward transfer).
- O-LoRA, EMNLP Findings 2023 (arXiv 2310.14152): orthogonal subspace learning, but LoRA-based, for sequential tasks.

These need an "old-task subspace." For a single pretrained diffusion base there is no task sequence, so that subspace must be approximated, either from the weight SVD (which is exactly the full-FT-OPLoRA route) or from activation covariance on a calibration set.

## Diffusion-specific alternative: Orthogonal Finetuning (OFT/COFT)

Source: "Controlling Text-to-Image Diffusion by Orthogonal Finetuning", Qiu et al., NeurIPS 2023 (arXiv 2306.07280).

OFT reparameterizes a weight as `R W` with `R` orthogonal (block-diagonal for parameter efficiency) and provably preserves hyperspherical energy, the pairwise angles between neurons, which the authors argue is what protects a T2I diffusion model's semantic generation ability. COFT adds a radius constraint for stability. This is a structurally different "preserve the pretrained structure" method built for diffusion, and it adapts more than a low-rank delta without being unconstrained full FT. Good candidate when the goal is "adapt while keeping base semantics."

## Practical ranking for this repo (full model, diffusion)

1. Replay (already in `anti-forgetting.md`): data-space, simplest and most robust, composes with everything. Best ROI.
2. Rank-1 EWC (already in `anti-forgetting.md`, full-FT only): cheap parameter-space soft anchor (rank-1 Fisher, one inner product per step). Good default for "stay near base."
3. Output distillation (referenced as distillation.md): output-space soft pull toward the frozen base; strong for diffusion because it directly preserves the denoising function; costs a teacher forward.
4. Full-FT orthogonal/null-space projection (GPM / Adam-NSCL / weight-SVD OPLoRA port): the genuine generalization of OPLoRA's hard guarantee, but heavier compute + memory and trickier to maintain under Adam. Worth it only when the hard subspace guarantee is specifically required.
5. OFT/COFT: diffusion-native provable-preservation alternative if a structured reparameterization is acceptable instead of unconstrained full FT.

Bottom line: OPLoRA cannot simply be switched on for full FT, but its core projection generalizes; for full-model diffusion the cheaper and better-proven path is Replay + Output Distillation with Rank-1 EWC as a parameter anchor, reserving a weight-space SVD projection (or OFT) for when a hard guarantee is the explicit requirement.

## Sources

- OPLoRA, arXiv 2510.13003 — https://arxiv.org/abs/2510.13003 , https://arxiv.org/html/2510.13003v2
- OPLoRA at AAAI 2026 — https://ojs.aaai.org/index.php/AAAI/article/view/40703
- O-LoRA, arXiv 2310.14152 — https://arxiv.org/abs/2310.14152
- GPM (Gradient Projection Memory), ICLR 2021 — https://arxiv.org/abs/2103.09762
- Adam-NSCL (null space continual learning), CVPR 2021 — https://arxiv.org/abs/2103.07113
- GNSP, arXiv 2507.19839 — https://arxiv.org/abs/2507.19839
- ROGO, arXiv 2301.12131 — https://arxiv.org/abs/2301.12131
- OFT (Orthogonal Finetuning), NeurIPS 2023, arXiv 2306.07280 — https://arxiv.org/abs/2306.07280
- LoRA-Null / MiLoRA (init-based, LoRA, full-weight-aware) — context only
