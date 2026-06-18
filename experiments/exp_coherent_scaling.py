"""
exp_coherent_scaling.py  --  THE paper-deciding experiment (Route C).

Question: a non-commuting coherent burst is more damaging than its Pauli twirl at
d=3 (ratio ~1.4-2.3). Does that coherent EXCESS persist or grow with code distance
d -- which defeats distance suppression (Lambda->1) -- or decay (small-code artifact,
suppression survives)?

This sweeps d for both a whole-patch and a spanning-disk non-commuting coherent burst
and records:
    p_L^coherent(d),  p_L^Pauli(d),  ratio(d),  and Lambda^coherent(d)/Lambda^Pauli(d).

RUNNING
  CPU (this sandbox / quick check):     python exp_coherent_scaling.py
  GPU (your box, d=5 exact):            COHSIM_BACKEND=cupy python exp_coherent_scaling.py
  d=7 (49 qubits) is beyond exact state vector (2^49); use the tensor-network backend
  (cuQuantum cuTensorNet / quimb) -- see docs/GPU_NOTES.md. The CoherentSim API is the
  oracle the tensor-network implementation must reproduce at d=3,5.

INTERPRETATION (state honestly in the paper)
  ratio(d) increasing or constant      -> coherent excess survives -> evidence for a
                                          coherent (non-Pauli) suppression floor (strong).
  ratio(d) -> 1 as d grows             -> discretization wins at scale -> suppression
                                          survives; coherence is a small-code effect.
  Either outcome is a publishable, falsifiable result.
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from coherent_sim import CoherentSim, _BK


def run(distances, thetas, shots_coh, shots_pauli, disk_mode="patch", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in distances:
        S = CoherentSim(d)
        if disk_mode == "patch":
            disk = list(range(S.n))
        else:  # spanning disk of radius 0.5*L_d about the centre
            c = S.coords.mean(0)
            Ld = (d - 1) * 2.0
            r2 = (S.coords[:, 0] - c[0]) ** 2 + (S.coords[:, 1] - c[1]) ** 2
            disk = list(np.where(r2 <= (0.5 * Ld) ** 2)[0])
        for t in thetas:
            c_ler = S.coherent_LER(disk, t, t, shots_coh, rng)
            p_ler = S.pauli_twirl_LER(disk, t, t, shots_pauli, rng)
            ratio = c_ler / p_ler if p_ler > 0 else float("nan")
            rows.append(dict(d=d, n=S.n, theta=t, disk=len(disk),
                             coherent_LER=c_ler, pauli_LER=p_ler, ratio=ratio))
            print(f"  d={d:2d} n={S.n:3d} theta={t:.2f} |disk|={len(disk):3d}  "
                  f"coh={c_ler:.4e}  pauli={p_ler:.4e}  ratio={ratio:.3f}")
    return rows


def main():
    print(f"backend = {_BK}")
    # d=3 runs here in seconds; d>=5 wants COHSIM_BACKEND=cupy on the GPU box.
    distances = [3, 5] if _BK == "cupy" else [3]
    thetas = [0.20, 0.30, 0.40, 0.50]
    print("\n=== whole-patch non-commuting coherent burst ===")
    rows = run(distances, thetas, shots_coh=4000, shots_pauli=120000, disk_mode="patch")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "coherent_scaling.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]
    print(f"\nsaved {out}")

    # ratio-vs-d summary at a fixed theta (the headline scaling)
    print("\nratio(d) at theta=0.30 (the headline number for the paper):")
    for r in rows:
        if abs(r["theta"] - 0.30) < 1e-9:
            print(f"  d={r['d']:2d}: ratio = {r['ratio']:.3f}")
    if _BK != "cupy":
        print("\n[only d=3 computed on CPU; set COHSIM_BACKEND=cupy on the GPU box for d=5]")


if __name__ == "__main__":
    main()
