# Findings log (validated, reproducible in this repo)

## F1 — pipeline correct on the real code
Rotated surface-code X/Z check matrices, logicals, and coords extracted from stim by
single-qubit error injection. Independent-Z code-capacity threshold ≈ 10% reproduced
(curves pinch at p≈0.10, blow up at p=0.12). Stabilizers commute; logicals anticommute.

## F2 — Paper 1 core, on the real surface code
One disk-burst geometry, q_d and b_d from the same decode:
- Fixed radius R=3 (localized): q_d grows 0.007→0.16, but b_d collapses 0.33→0.04→7e-4→0;
  p_L→0, Λ→∞. The code corrects a localized burst.
- Spanning radius R=0.5 L_d (ξ∝d): b_d persists ≈0.08; p_L grows; Λ→1 (from below).
Conclusion: the burst floor exists iff ξ_burst ≳ d. Theorem-5 area-scaling (localized) and
Theorem-6 b_d→b_0 (spanning) cannot co-hold for one localized burst. The cosmic-ray
mechanism (fixed phonon-spread length) is the localized column -> no floor at scale.

## F3 — Paper 2 seed (Route C), d=3
Non-commuting whole-patch coherent burst vs its Pauli twirl, separable MWPM decoder:
ratio(coherent/Pauli) = 2.5, 2.2, 1.8, 1.5 at theta = 0.20, 0.30, 0.40, 0.50.
=> coherence is NOT trivially discretized away; the Pauli model under-predicts at d=3.
The d=3 dense result is the correctness oracle for the scalable engine (reproduced exactly
by the cupy backend and the in-place batched rewrite). d-scaling answered at d=5 in F4.

## F4 — Paper 2, d=5 on GPU: the excess scaling is theta-dependent (a crossover)
Same observable at d=5 (25-qubit exact state vector), computed on 4x RTX 2000 Ada with the
in-place batched engine (`src/coherent_sim.py`) sharded across all 4 GPUs
(`experiments/run_d5_multigpu.py`, checkpointed). Coherent LER is a soft estimator
(per-shot (1-<Z_L>)/2), so a few thousand shots pin the ratio; errors below are 1 sigma.

  theta   d=3 ratio   d=5 ratio (+/- SE)     shots (coh / Pauli)   d=5 vs d=3
  0.20      2.5        3.41 +/- 0.16          6000 / 120000         +5.8 sigma  (GROWS)
  0.30      2.2        2.32 +/- 0.19           300 / 120000         +0.6 sigma  (flat)
  0.40      1.8        1.63 +/- 0.08           300 / 120000         -2.1 sigma  (shrinks)
  0.50      1.5        1.31 +/- 0.05           300 / 120000         -4.2 sigma  (shrinks)
  (results/coherent_scaling_d5_multigpu.csv + ..._t020_6k.csv)

Reading: the coherent excess over the Pauli twirl is NOT a single verdict with d. It GROWS
from d=3 to d=5 at small rotation angle (theta=0.20, the weak-burst / physically relevant
regime, 2.5 -> 3.41) and SHRINKS below the d=3 value at large angle (theta=0.40, 0.50),
where discretization wins at scale. Crossover near theta=0.30. So the "small-code artifact"
story holds only for strong bursts; for weak coherent errors the excess survives and grows
over this range -- the direction that, if it continued, points at a coherent (non-Pauli)
contribution to the floor. The excess everywhere stays > 1 (1.3-3.4): the Pauli model
under-predicts at d=5, not just d=3.

OPEN (do not over-read F4):
- Two distances only (d=3, d=5). d=7 (49 qubits, tensor-network backend; needs cuQuantum,
  which has no Windows wheel on this box) would show whether the small-theta growth is
  monotonic or a two-point artifact.
- Separable MWPM decoder throughout. A correlated / coherent-aware decoder may close the
  small-theta gap (the standing escape) -- still untested.
- Whole-patch burst; the spanning-disk geometry at d=5 is not yet swept here.

## Honest scope
All conditional and falsifiable. No statement about real hardware is drawn from
simulation. d=5 resolves the d=3->d=5 trend per F4 but does not settle the asymptotic
d-scaling; d=7 and the correlated-decoder test are the remaining computations.
