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
OPEN (needs GPU d=5,7): does ratio(d) persist/grow (coherent floor, new physics) or
decay (small-code artifact)? Also OPEN: does a correlated decoder close the gap?

## Honest scope
All conditional and falsifiable. No statement about real hardware is drawn from
simulation. d=3 cannot settle d-scaling; that is the next computation on the GPU box.
