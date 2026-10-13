"""tools/sdpa_probe.py compares attention kernels with a float32 reference (CPU: math and flash)."""

from test.test_iris import ROOT  # noqa: F401  (puts the repo on sys.path)
from tools import sdpa_probe


def test_kernels_pass_on_cpu(capsys):
    code = sdpa_probe.main(['--device', 'cpu', '--batch-size', '2', '--tokens', '70', '--heads', '4',
                            '--kv-heads', '2', '--head-dim', '32', '--backends', 'default', 'math'])
    out = capsys.readouterr().out
    assert code == 0, out
    assert 'math      ok' in out and 'default uses:' in out


def test_a_wrong_kernel_fails(monkeypatch, capsys):
    real = sdpa_probe.run

    def broken(q, k, v, grad, backend):
        out, dq, dk, dv = real(q, k, v, grad, backend)
        dq = dq.clone()
        dq[0, 0, 0, 0] = float('nan')
        return out, dq, dk, dv
    monkeypatch.setattr(sdpa_probe, 'run', broken)
    code = sdpa_probe.main(['--device', 'cpu', '--batch-size', '1', '--tokens', '40', '--heads', '2',
                            '--kv-heads', '1', '--head-dim', '16', '--gain', '1', '--backends', 'math'])
    out = capsys.readouterr().out
    assert code == 1
    assert 'dq 1 non-finite' in out and 'math FAILS at 40 tokens' in out
