# GPU / scale-up notes (4x NVIDIA, 512 GB, 256 cores)

## Backend switch
`src/coherent_sim.py` reads `COHSIM_BACKEND`. Default `numpy` (CPU). Set `cupy` to put
the state vector and all gate/measurement ops on GPU with no code change:
```
COHSIM_BACKEND=cupy python experiments/exp_coherent_scaling.py
```
Single-qubit gates are applied by tensor reshaping (no 2^n x 2^n matrices), so memory is
just the state vector: 2^n complex128.

## Exact state-vector reach
| d  | data qubits n | state vector (complex128) | where |
|----|---------------|---------------------------|-------|
| 3  | 9             | 8 KB                      | CPU (oracle) |
| 5  | 25            | 0.5 GB                    | one GPU, exact |
| 7  | 49            | 9 PB                      | INFEASIBLE exact -> tensor network |

Exact GPU is fine through d=5. Use complex64 to halve memory if needed (check the d=3
oracle still matches first).

## d=7 and beyond: tensor networks
49 qubits is past exact. Two routes, both reproducing the d=3,5 exact numbers as a check:
1. **cuQuantum cuStateVec** handles up to ~32 qubits on multi-GPU with distribution
   (still short of 49) — useful for d=5 large-shot runs and partial d=7 patches.
2. **cuTensorNet / quimb MPS-PEPS** for d=7: represent the surface-code patch state and
   the (low-depth) burst + stabilizer-measurement as a tensor network; contract on GPU.
   The burst is a single layer of 1-qubit rotations and the syndrome is one round of
   stabilizer projections, so bond dimensions stay modest for the memory experiment.
   `pip install cuquantum-python quimb opt_einsum`.

## Parallelism
- Coherent trajectories are embarrassingly parallel: shard `shots` across the 4 GPUs
  and/or 256 cores, each with an independent RNG seed; sum the soft logical-error
  estimates. Variance scales as 1/total_shots.
- The Pauli phase-diagram sweep (`exp_surface_burst.py` style) is pure CPU MWPM and
  scales linearly across all 256 cores via `multiprocessing` over (d, xi, sigma) points.

## Validation gate before any production run
Run d=3 on the target backend and confirm the coherent/Pauli ratios match the CPU oracle
(~2.5, 2.2, 1.8, 1.5 at theta=0.20,0.30,0.40,0.50). Only then scale to d=5,7.
