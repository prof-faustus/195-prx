# Research Program — Common-Mode Noise and the Limits of Distance Suppression

**Program thesis (conditional, falsifiable — NOT an impossibility claim):**
The exponential distance-suppression of logical error in a surface-code memory is
*conditional*. It can be driven to a floor (Λ(d)→1) only by noise that is (i) correlated
on a length that scales with the code, ξ_corr(d) ≳ d, and (ii) not reducible to a
below-threshold Pauli channel by syndrome extraction. We make each condition precise,
prove the conditional theorems, test them on the real rotated surface code, and state
exactly what would falsify them. Whether real hardware satisfies the conditions is left
as the stated open empirical question.

This is a program, not a paper. The papers below are self-contained and submittable
regardless of how the later, harder ones resolve.

---

## Paper 1 — Conditional suppression-collapse theorem + real-code instrument
**Target:** PRA or PRX Quantum. **Status:** results in hand; ready to write.

Claim: p_L(d) ≥ q_d b_d, and Λ(d)→1 iff q_d b_d is subexponential and dominates the local
branch. The decisive correction to the burst literature: an area-scaling burst *rate*
(q_d ~ d²) does **not** by itself produce a floor, because for a **localized** burst the
conditional failure b_d collapses (the code corrects it). On the real surface code, a
fixed-radius burst gives b_d → 0 (Λ → ∞, corrected); only a burst whose damaged region
scales with the code, ξ_burst ≳ d, gives b_d → b_0 > 0 (Λ → 1). The entire mechanism
therefore reduces to the single physical condition ξ_burst(d) ≳ d.
Instrument: `experiments/exp_surface_burst.py` (validated; threshold ~10% reproduced).
Falsification conditions are explicit (fixed ξ → ξ/d → 0 → no floor).

## Paper 2 — Coherent (non-Pauli) bursts and error discretization (Route C)
**Target:** PRL if the scaling is sharp, else PRA. **Status:** d=3 result in hand; d=5/7
on GPU is the open computation.

Open question: a non-commuting coherent burst is more damaging than its Pauli twirl
(ratio ~1.4–2.3 at d=3 under a separable decoder). Does that coherent **excess** persist
or grow with d (→ a coherent suppression floor, new physics: the missing piece
Burgelman–Viola opened) or decay (→ small-code artifact, suppression survives)?
Instrument: `src/coherent_sim.py` (GPU-ready state vector) + `experiments/exp_coherent_scaling.py`.
Must also test whether a **correlated** decoder closes the gap (a standing escape).

## Paper 3 — Deriving ξ_burst(d) from device physics
**Target:** PRA / PRApplied. **Status:** not started.

Solve phonon/quasiparticle transport + generation on a realistic chip geometry to obtain
the *actual* correlation length of the induced error field, and decide: fixed physical
length (→ ξ/d → 0 → suppression survives, cosmic rays do not floor a large code) or
growing (→ floor). Converts Paper 1's load-bearing assumption into a derived quantity.

## Paper 4+ — Extensions
Correlated/coherent-aware decoders; circuit-level + multi-round; leakage/erasure channel
(higher per-event damage, different threshold); other code families (color, qLDPC).

---

## Hard rules for this program
- Every claim is conditional and falsifiable; the impossibility framing is **not** used.
- Code is validated at small d on CPU before any GPU production run; the dense d=3 result
  is the correctness oracle for the scalable engine and any tensor-network backend.
- The Pauli model is the **conservative** baseline only if Paper 2 shows the coherent
  excess decays; if it persists, the Pauli baseline under-predicts and must not be used.
- No physical claim about real hardware is made from simulation alone.
