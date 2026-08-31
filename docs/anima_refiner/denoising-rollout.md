# Denoising rollout

An optional distillation objective that measures agreement where it actually matters: at the
frozen DiT's own prediction, on inputs the DiT itself produced.

**Off by default.** At `loss_weight = 0` the DiT is discarded exactly as it always was, nothing
extra is allocated, and this stage behaves as it did before the option existed.

## Why the project needs it

Distillation's default objective pushes the teacher's and the student's text features through
the DiT's frozen cross-attention with a set of fixed random probe queries, and compares the
outputs. Cross-attention output is a weighted sum over text positions, which is exactly what
makes the comparison legal across two different tokenizations — the teacher's sequence is
indexed by T5 tokens, the student's by Qwen BPE.

That same property bounds what the objective can see. What it compares are sums over positions,
and a sum does not care which token holds which content. A student that smears the same average
content across every position satisfies it about as well as one that puts the right content in
the right place.

This is the documented failure mode for naive text-encoder distillation.
[Scaling Down Text Encoders of Text-to-Image Diffusion Models (CVPR 2025)](https://arxiv.org/html/2503.19897v1)
reports the student's embedding space collapsing, with "rat", "cat" and "man" returning
identical embeddings. Their remedy is to stop comparing embeddings and start comparing what the
diffusion model *predicts* from them. That is what this implements.

`relational_loss_weight` (on by default, free) already prices collapse directly by matching the
teacher's pairwise distance structure, and the `spread` figure in the progress bar makes it
visible. **Enable the rollout when `spread` shows collapse happening anyway.** It is the
stronger objective and the more expensive one.

## How it works

```
x ← pure Gaussian noise                            no images, no VAE
for each step, t from 1 down toward 0:
    v ← DiT(x, t, teacher_features)                teacher drives the path
    remember (x, t)
    x ← x − Δt · v                                 Euler step
                                                    ── all of the above under no_grad ──

pick `loss_points` of the remembered (x, t):
    target ← DiT(x, t, teacher_features)           no_grad
    pred   ← DiT(x, t, student_features)           with gradient
    loss   += ‖pred − target‖²
```

### What the teacher does, and what the student does

The teacher does two things: it produces the text features being matched, and it drives the
trajectory. The student does one: it produces the text features being compared against them.

**The student never advances the trajectory and never sees its own output as input.** This is
not a simplification — it is what Algorithm 1 of the paper does, and it has a consequence worth
stating plainly: there is no error accumulation, and none of the exposure-bias literature that
the word "rollout" usually invokes applies here. Those results are about a student rolling out
its own predictions. This one does not.

### Which timesteps

All of them, in the sense that the trajectory covers `t = 1` down toward `0` in `steps` even
increments; and `loss_points` of them per training step, drawn at random. Across a run the
random draw covers the whole path.

`t = 1` is pure noise and `t = 0` is clean, which is the rectified-flow convention Anima was
trained with. The paper is written for DDPM, so its posterior update does not carry over — the
Euler step above replaces it. This is the one place where the method had to be adapted rather
than transcribed.

### Why a rollout at all, rather than random noisy latents

Because this stage has no images. Ordinary diffusion training builds `x_t = (1−t)·x_0 + t·noise`
from a real `x_0`; there is no `x_0` here, and adding one would mean loading the VAE and a
dataset of images, which is precisely what this stage exists to avoid. The trajectory is the
only way to obtain `x_t` that lie where the model will actually be asked to predict.

### Gradient flow

Only the student's forward carries gradient. The trajectory and the teacher's predictions are
computed under `no_grad`, so:

- the cost of `steps` is inference cost, not backward cost,
- `steps` and `loss_points` are independent knobs,
- lengthening the trajectory is cheap; measuring at more points is not.

There is a test asserting the trajectory tensors do not require grad, because if that ever
changes the two knobs stop being independent and the memory profile changes silently.

## Configuration

Everything lives under a new `[rollout]` table in the distillation config. No existing key was
renamed, removed, or given a new meaning.

| Key | Default | Meaning |
| --- | --- | --- |
| `loss_weight` | `0.0` | Weight of the rollout term. **0 disables the feature entirely**, including keeping the DiT resident. |
| `steps` | `8` | Points along the trajectory. Cheap — the walk is `no_grad`. |
| `loss_points` | `2` | How many of them the loss is measured at. The expensive knob. |
| `resolution` | `256` | Pixel resolution the latent is sized from. Must be a multiple of 16. |
| `guidance_scale` | `0.0` | Classifier-free guidance during the rollout. 0 evaluates only the conditional branch. |

```toml
[rollout]
loss_weight = 1.0
steps = 8
loss_points = 2
resolution = 256
guidance_scale = 0.0
```

### Choosing `loss_points`

Measured on a reduced-scale replica (2-block DiT, 16×16 latent, CPU), forward plus backward:

| `loss_points` | time | relative |
| --- | --- | --- |
| 1 | 26 ms | 1.0× |
| 2 | 49 ms | 1.9× |
| 4 | 97 ms | 3.7× |
| 8 | 197 ms | 7.6× |

The 8-step trajectory itself costs 78 ms, a fixed price roughly equal to three loss points. Cost
in `loss_points` is linear, and adjacent trajectory points give correlated signal — the same
argument `[probe] num_blocks` rests on. Two is the knee.

### `resolution`

Must be a multiple of 16: the VAE downsamples by 8 and the DiT patches by 2, so anything else
leaves a fractional patch grid. `256` gives a 32×32 latent and 256 tokens after patching. It is
validated at startup rather than failing inside the first forward.

### `guidance_scale`

Above 0, the unconditional branch is evaluated on both sides, doubling the DiT forwards. It is
worth the cost when you care about CFG sampling specifically: the unconditional branch is fed by
an empty caption, and an empty caption is the one input where the refiner path diverges most
from the T5-native DiT it feeds. The student's unconditional features are recomputed with
gradient every step, because they depend on the refiner too.

The paper samples guidance between 2 and 5. A fixed value is used here; sweeping it is not
implemented.

## Cost

The DiT stays resident when the feature is on. For Anima that is roughly **3.5–4 GB in bf16**
(estimated from 28 blocks at 2048 channels; not measured against a real checkpoint), taking this
stage from about 6 GB to **10–12 GB**, plus activations for the forwards.

That is the entire reason it is opt-in. The ~6 GB figure is what makes this stage runnable on
modest hardware, and it should not quietly stop being true.

## Single-GPU and multi-GPU

**Single GPU.** Nothing special: the DiT is one more frozen module on the same device.

**Multi-GPU.** The DiT is frozen and is not part of the module DDP or ZeRO wraps — only the
refiner is — so it contributes no gradients to reduce and no parameters to shard. It is
replicated per rank, which is what makes the memory cost per-GPU rather than shared.

The noise generator is **rank-offset** (`seed + 10000 + rank`), so each rank walks a different
trajectory and the effective number of trajectories scales with world size. This mirrors what
the caption stream already does (`random.seed(seed + rank)`), and is deliberately unlike the
probe queries, which are shared across ranks so that every rank optimises the same objective.

**Checkpoint and resume.** The rollout adds no state. Turning it on or off between runs is safe;
the refiner file and `distill_state.pt` are unaffected either way.

## Limitations and known issues

- **Never run on a GPU.** Everything here is verified on CPU with a small real `MiniTrainDIT`.
  No multi-rank run, no full-size run, no image produced.
- **The memory figure is an estimate**, derived from the DiT's shape rather than measured.
- **`guidance_scale` is fixed**, not sampled per step as the paper does.
- **The trajectory is uninformed by the caption distribution.** It starts from Gaussian noise
  every step; nothing biases it toward the region of latent space real images occupy.
- **`steps` interacts with nothing else.** Longer trajectories reach lower `t` (cleaner
  latents); with `steps = 8` the lowest point visited is `t = 0.125`, so the model is never
  compared very close to `t = 0`. Raise `steps` if the late trajectory matters for your use.
