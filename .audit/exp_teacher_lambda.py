"""Design evidence: what lambda means in the proposed teacher-guided loss.

    python .audit/exp_teacher_lambda.py

Backs the numbers quoted in docs/anima_refiner/teacher-guided-training.md. Nothing here touches
the repo's code -- the proposal is not implemented -- it settles one question about the loss
shape before anybody writes it:

    L = (1-lam) * D(v_s, v_gt) + lam * D(v_s, v_T)

Does the large gap between the two loss VALUES mean lam is not really the mixing weight?

For squared error, no. Mixing the two losses is algebraically the same as regressing onto the
mixed target (1-lam)*v_gt + lam*v_T, because the v_s terms collect. The gap in the values is the
irreducible residual of the ground-truth sample, which is zero-mean: it inflates the number in
the log and contributes nothing to the expected gradient. Measured below as a cosine against the
mixed-target reference, which comes out at 0.9999+ everywhere.

What the gap does cost is gradient noise, and that is the mechanism the whole feature runs on:
the per-draw gradient deviation grows to the size of the signal itself at high t, and the
teacher is a pre-averaged target that does not have it.

Huber and smooth-L1 are the exception. They clip large residuals, so the ground-truth term
saturates at high t while the teacher term does not, and lam drifts from its nominal value.

MODEL, and its one assumption. v_gt = vbar + residual, with the residual's scale taken to grow
with t. That is the manifold argument in the doc -- at low t a real latent lies near the data
manifold, so both x_0 and the noise are nearly recoverable, while at high t x_0 is not -- and it
is an assumption here, not a measurement on real latents. The cosine result does NOT depend on
it: the target-blend identity is algebra and holds for any residual. Only the noise column and
the Huber drift move with it.
"""
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

DIM = 4096
DRAWS = 800
LAM = 0.5
HUBER_DELTA = 1.0

vbar = torch.randn(DIM)                   # E[v | x_t, caption]: what the student should output
v_teacher = vbar + 0.15 * torch.randn(DIM)   # an expectation already, plus the 0.6B encoder's bias
v_student = vbar + 0.40 * torch.randn(DIM)   # the student, currently wrong


def grad_mse(pred, target):
    return 2 * (pred - target)


def grad_huber(pred, target, delta=HUBER_DELTA):
    r = pred - target
    return torch.where(r.abs() <= delta, r, delta * r.sign())


def cosine(a, b):
    return (a @ b / (a.norm() * b.norm())).item()


def implied_lambda(expected_grad):
    """The target-blend weight whose gradient best matches this one.

    For MSE this recovers LAM exactly. For Huber it does not, which is the point.
    """
    best_cos, best_lam = None, None
    for candidate in torch.linspace(0, 1, 2001):
        reference = grad_mse(v_student, (1 - candidate) * vbar + candidate * v_teacher)
        c = cosine(expected_grad, reference)
        if best_cos is None or c > best_cos:
            best_cos, best_lam = c, candidate.item()
    return best_lam


def main():
    reference = grad_mse(v_student, (1 - LAM) * vbar + LAM * v_teacher)

    print(f'lambda = {LAM}, huber_delta = {HUBER_DELTA}, {DRAWS} residual draws per row\n')
    print(f"{'t':>5} {'L_gt':>8} {'L_T':>8} {'ratio':>7} "
          f"{'MSE cos':>10} {'noise/signal':>13} {'Huber lam':>10}")

    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        residual_std = t  # the one assumption; see the module docstring

        mse_grads, huber_grads, gt_losses = [], [], []
        for _ in range(DRAWS):
            v_gt = vbar + residual_std * torch.randn(DIM)
            mse_grads.append((1 - LAM) * grad_mse(v_student, v_gt)
                             + LAM * grad_mse(v_student, v_teacher))
            huber_grads.append((1 - LAM) * grad_huber(v_student, v_gt)
                               + LAM * grad_huber(v_student, v_teacher))
            gt_losses.append(((v_student - v_gt) ** 2).mean())

        mse_grads = torch.stack(mse_grads)
        expected_mse = mse_grads.mean(0)
        expected_huber = torch.stack(huber_grads).mean(0)

        # How far a single draw's gradient sits from the mean, relative to the mean.
        noise = (mse_grads - expected_mse).norm(dim=1).mean().item() / expected_mse.norm().item()

        loss_gt = torch.stack(gt_losses).mean().item()
        loss_teacher = ((v_student - v_teacher) ** 2).mean().item()

        print(f'{t:5.2f} {loss_gt:8.3f} {loss_teacher:8.3f} {loss_gt / loss_teacher:7.2f} '
              f'{cosine(expected_mse, reference):10.6f} {noise:13.1f} '
              f'{implied_lambda(expected_huber):10.3f}')

    print('\nMSE cos ~ 1.0 everywhere: lambda is exactly the target-mixing weight, and the '
          '\nratio column is zero-mean residual, not a distortion. Do not normalise the terms.'
          '\nHuber lam drifts away from 0.5 as t rises: clipping suppresses the noisy '
          '\nground-truth term, so lambda is approximate when huber_delta is set.')


if __name__ == '__main__':
    main()
