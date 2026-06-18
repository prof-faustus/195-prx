"""
Baseline validation on the REAL code: rotated surface-code memory under
independent circuit-level depolarizing noise, decoded with MWPM.
Confirms (i) the stim+pymatching pipeline is correct and
(ii) the surface code shows clean exponential distance-suppression (Lambda>1)
below threshold -- the behaviour the common-bath mechanism must destroy.
"""
import numpy as np
import stim
import pymatching


def logical_error_rate(d, p, shots):
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=d,
        distance=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    dem = circ.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circ.compile_detector_sampler()
    syndrome, actual = sampler.sample(shots, separate_observables=True)
    pred = matching.decode_batch(syndrome)
    fails = int(np.sum(np.any(pred != actual, axis=1)))
    pL = fails / shots
    se = np.sqrt(max(pL * (1 - pL), 1e-12) / shots)
    return pL, se


def run():
    distances = [3, 5, 7, 9, 11]
    shots = 200_000
    print("Rotated surface-code memory, independent circuit-level depolarizing noise.")
    for p in [0.001, 0.003, 0.005]:
        print(f"\n# physical error rate p = {p}")
        pls = []
        for d in distances:
            pL, se = logical_error_rate(d, p, shots)
            pls.append(pL)
            print(f"  d={d:2d}   p_L(per {d}-round shot) = {pL:.5e} +/- {se:.1e}")
        print("  Lambda(d)=p_L(d)/p_L(d+2):", end="  ")
        for k in range(len(distances) - 1):
            lam = pls[k] / pls[k + 1] if pls[k + 1] > 0 else float('inf')
            print(f"{distances[k]}->{distances[k+1]}: {lam:.2f}", end="   ")
        print()


if __name__ == "__main__":
    run()
