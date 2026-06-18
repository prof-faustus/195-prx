"""
Refined surface-code burst study + figures.

Measures, on the REAL rotated surface code:
  (A) local-only p_L(d) under benign independent background  -> exponential reference
  (B) conditional burst failure b_d(d) measured DIRECTLY (a burst is forced each
      shot) for two burst geometries that differ ONLY in how the damaged disk
      scales with code distance:
        - FIXED radius R = R0           (localized; the cosmic-ray case)
        - SPANNING radius R = c * L_d   (correlation length grows with the patch)
  (C) the resulting Lambda(d) for the physical mixture
        p_L(d) = (1 - q_d) p_local(d) + q_d b_d(d),   q_d from a Poisson area model.

Conclusion this is designed to expose (honestly, either way):
  b_d collapses for a fixed-size burst (so Lambda survives) and only persists when
  the damaged region scales with the code -- isolating xi_burst >~ d as the load-
  bearing physical condition.
"""
import numpy as np
import stim
import pymatching
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(2024)
PITCH = 2.0


def build_surface_check(d):
    circ = stim.Circuit.generated("surface_code:rotated_memory_x", rounds=1, distance=d)
    data = [t.value for t in next(i for i in circ if i.name == "RX").targets_copy()]
    cmap = {}
    for inst in circ:
        if inst.name == "QUBIT_COORDS":
            a = inst.gate_args_copy()
            cmap[inst.targets_copy()[0].value] = (a[0], a[1])
    base = list(circ)

    def col_for(q):
        new = stim.Circuit()
        placed = False
        for inst in base:
            new.append(inst)
            if inst.name == "R" and not placed:
                new.append("Z_ERROR", [q], 0.01)
                placed = True
        dem = new.detector_error_model(decompose_errors=False, flatten_loops=True)
        dets, obs = [], []
        for e in dem:
            if e.type == "error":
                dets = [t.val for t in e.targets_copy() if t.is_relative_detector_id()]
                obs = [t.val for t in e.targets_copy() if t.is_logical_observable_id()]
        return dets, obs

    ndet = circ.num_detectors
    H = np.zeros((ndet, len(data)), dtype=np.uint8)
    L = np.zeros((max(1, circ.num_observables), len(data)), dtype=np.uint8)
    for idx, q in enumerate(data):
        dets, obs = col_for(q)
        for dd in dets:
            H[dd, idx] = 1
        for oo in obs:
            L[oo, idx] = 1
    Ht = H[H.sum(1) > 0]
    coords = np.array([cmap[q] for q in data], dtype=float)
    return Ht, L, coords


def ler(M, Ht, L, E):
    syn = (Ht @ E.T % 2).T.astype(np.uint8)
    pred = M.decode_batch(syn)
    actual = (L @ E.T % 2).T.astype(np.uint8)
    f = int(np.sum(np.any(pred != actual, axis=1)))
    return f / E.shape[0]


def local_only(M, Ht, L, coords, p0, shots):
    E = (rng.random((shots, Ht.shape[1])) < p0).astype(np.uint8)
    return ler(M, Ht, L, E)


def conditional_burst(M, Ht, L, coords, p0, p_b, R, shots):
    """b_d: logical failure probability GIVEN a burst is present (forced each shot)."""
    n = coords.shape[0]
    xmin, ymin = coords.min(0); xmax, ymax = coords.max(0)
    cx = rng.uniform(xmin, xmax, shots); cy = rng.uniform(ymin, ymax, shots)
    d2 = (coords[None, :, 0] - cx[:, None]) ** 2 + (coords[None, :, 1] - cy[:, None]) ** 2
    inside = d2 <= R * R
    P = np.where(inside, p_b, p0)
    E = (rng.random((shots, n)) < P).astype(np.uint8)
    return ler(M, Ht, L, E)


def q_area(coords, area_rate):
    xmin, ymin = coords.min(0); xmax, ymax = coords.max(0)
    area = (xmax - xmin + PITCH) * (ymax - ymin + PITCH)
    return 1.0 - np.exp(-area_rate * area)


def main():
    distances = [3, 5, 7, 9, 11, 13, 15]
    print("building codes (deterministic-injection extraction)...")
    cache = {}
    for d in distances:
        cache[d] = build_surface_check(d)
        print(f"  d={d}: data={cache[d][0].shape[1]}, X-stabs={cache[d][0].shape[0]}")

    p0 = 0.001
    p_b = 0.40
    area_rate = 2.0e-4
    R0 = 3.0
    cspan = 0.5
    shots_local = 400000
    shots_b = 120000

    rows = []
    print("\nmeasuring...")
    for d in distances:
        Ht, L, coords = cache[d]
        M = pymatching.Matching(Ht, faults_matrix=L)
        Ld = (d - 1) * PITCH
        pl = local_only(M, Ht, L, coords, p0, shots_local)
        b_fixed = conditional_burst(M, Ht, L, coords, p0, p_b, R0, shots_b)
        b_span = conditional_burst(M, Ht, L, coords, p0, p_b, cspan * Ld, shots_b)
        q = q_area(coords, area_rate)
        pL_fixed = (1 - q) * pl + q * b_fixed
        pL_span = (1 - q) * pl + q * b_span
        rows.append(dict(d=d, Ld=Ld, p_local=pl, q=q,
                         b_fixed=b_fixed, b_span=b_span,
                         pL_fixed=pL_fixed, pL_span=pL_span))
        print(f"  d={d:2d}  p_local={pl:.3e}  q={q:.4f}  "
              f"b_fixed={b_fixed:.3e}  b_span={b_span:.3e}  "
              f"pL_fixed={pL_fixed:.3e}  pL_span={pL_span:.3e}")

    def lam(key):
        return [rows[k][key] / rows[k + 1][key] if rows[k + 1][key] > 0 else np.nan
                for k in range(len(rows) - 1)]
    print("\nLambda(d):")
    print("  local-only :", [f"{x:.2f}" for x in lam('p_local')])
    print("  fixed burst:", [f"{x:.2f}" for x in lam('pL_fixed')])
    print("  span  burst:", [f"{x:.2f}" for x in lam('pL_span')])

    # save CSV
    import csv
    with open("surface_burst_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(r) for r in rows]

    ds = [r['d'] for r in rows]
    # ---- Figure 1: b_d(d) fixed vs spanning ----
    plt.figure(figsize=(6, 4.2))
    plt.semilogy(ds, [r['b_fixed'] for r in rows], 'o-', label='fixed radius $R=3$ (localized)')
    plt.semilogy(ds, [max(r['b_span'], 1e-6) for r in rows], 's-', label='spanning radius $R=0.5\\,L_d$')
    plt.xlabel('code distance $d$'); plt.ylabel('conditional burst failure $b_d$')
    plt.title('Conditional logical failure given a burst (real surface code)')
    plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
    plt.savefig('fig_bd.png', dpi=140)

    # ---- Figure 2: p_L(d) ----
    plt.figure(figsize=(6, 4.2))
    plt.semilogy(ds, [max(r['p_local'], 1e-7) for r in rows], '^-', label='local only (no burst)')
    plt.semilogy(ds, [max(r['pL_fixed'], 1e-7) for r in rows], 'o-', label='fixed-radius burst')
    plt.semilogy(ds, [max(r['pL_span'], 1e-7) for r in rows], 's-', label='spanning burst')
    plt.xlabel('code distance $d$'); plt.ylabel('logical error per cycle $p_L$')
    plt.title('Distance scaling: localized burst is corrected, spanning burst floors')
    plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
    plt.savefig('fig_pL.png', dpi=140)

    print("\nsaved surface_burst_results.csv, fig_bd.png, fig_pL.png")


if __name__ == "__main__":
    main()
