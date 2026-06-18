# 195 PRX — Common-Mode Noise & Distance Suppression

Conditional, falsifiable study of when surface-code distance-suppression collapses.
See `PROGRAM.md` for the multi-paper roadmap and `docs/FINDINGS.md` for what is
established. This is theory + simulation; no claim about real hardware is made from
simulation alone.

## Layout
- `src/coherent_sim.py` — GPU-ready coherent-burst state-vector simulator (numpy/cupy).
- `experiments/exp_independent_validation.py` — surface-code threshold sanity (~10%).
- `experiments/exp_rep_correlation.py` — repetition-code: correlation length ξ≳d collapses Λ at fixed error rate.
- `experiments/exp_surface_burst.py` — real surface code: localized burst corrected, spanning burst floors (Paper 1).
- `experiments/exp_coherent_scaling.py` — coherent vs Pauli-twirl LER scaling with d (Paper 2 / Route C).
- `results/` — output CSVs and figures.

## Run (CPU, validation)
```
pip install -r requirements.txt
python experiments/exp_independent_validation.py
python experiments/exp_surface_burst.py
python experiments/exp_coherent_scaling.py        # d=3 only on CPU
```

## Run (GPU, production) on the 4-GPU box
```
pip install cupy-cuda12x cuquantum-python quimb opt_einsum
COHSIM_BACKEND=cupy python experiments/exp_coherent_scaling.py   # adds d=5 exact
```
d=7 (49 qubits) exceeds exact state vector (2^49). Use the tensor-network backend
(see `docs/GPU_NOTES.md`); the d=3,5 exact numbers are the oracle it must reproduce.

## Validated so far (all reproduced in this repo)
- Surface-code code-capacity threshold ≈ 10% (independent Z, MWPM).
- Fixed-radius burst: b_d → 0, Λ → ∞ (corrected). Spanning burst (ξ∝d): Λ → 1.
- Non-commuting coherent burst at d=3: LER 1.4–2.3× the Pauli twirl (separable decoder).
