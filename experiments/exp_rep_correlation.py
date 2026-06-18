"""
Repetition-code test of the common-bath mechanism.

Physical model (derived in the analytic note):
  A Gaussian dephasing field phi_i acts on each data qubit i with
      <phi_i> = 0,  Cov(phi_i, phi_j) = sigma^2 * exp(-|x_i - x_j| / xi).
  After syndrome extraction the coherent rotation exp(-i phi_i Z_i) is
  projected to a Pauli flip on qubit i with conditional probability sin^2(phi_i),
  qubits conditionally independent GIVEN the field.

Key controlled-comparison fact:
  The per-qubit MARGINAL flip probability is
      pbar = E[sin^2(phi_i)] = (1 - exp(-2 sigma^2)) / 2,
  which depends ONLY on sigma (the marginal variance C_ii = sigma^2),
  NOT on xi.  So we can sweep the correlation length xi at FIXED physical
  error rate pbar, and ask the single clean question:

      does the distance-suppression factor Lambda survive
      as correlations get long-ranged, at the same error rate?

Decoder: MWPM (PyMatching) on the standard repetition-code matching graph.
"""

import numpy as np
import scipy.linalg
import pymatching
import scipy.sparse as sp

rng = np.random.default_rng(20260617)


def rep_code_matrices(d):
    """Distance-d repetition code: d data qubits, d-1 checks.
    H[j,j]=H[j,j+1]=1.  Logical observable = qubit 0 (PyMatching rep-code convention)."""
    rows, cols = [], []
    for j in range(d - 1):
        rows += [j, j]
        cols += [j, j + 1]
    H = sp.csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(d - 1, d))
    L = sp.csc_matrix((np.array([1], dtype=np.uint8), (np.array([0]), np.array([0]))), shape=(1, d))
    return H, L


def marginal_pbar(sigma):
    return 0.5 * (1.0 - np.exp(-2.0 * sigma**2))


def sample_logical_error_rate(d, sigma, xi, n_shots, batch=200000):
    """Monte-Carlo logical error rate for distance d, dephasing std sigma,
    correlation length xi, using correlated Gaussian field + sin^2 projection."""
    H, L = rep_code_matrices(d)
    matching = pymatching.Matching(H, faults_matrix=L)

    x = np.arange(d, dtype=float)
    # covariance with exponential spatial kernel
    C = sigma**2 * np.exp(-np.abs(x[:, None] - x[None, :]) / xi)
    C = C + 1e-9 * np.eye(d)            # jitter for Cholesky stability at large xi
    Lchol = np.linalg.cholesky(C)       # lower-triangular

    fails = 0
    done = 0
    while done < n_shots:
        m = min(batch, n_shots - done)
        z = rng.standard_normal((m, d))
        phi = z @ Lchol.T               # (m, d) correlated Gaussian field
        p = np.sin(phi) ** 2            # conditional flip prob per qubit
        e = (rng.random((m, d)) < p).astype(np.uint8)
        syn = (H.dot(e.T) % 2).T.astype(np.uint8)   # (m, d-1) syndromes
        pred = matching.decode_batch(syn)               # (m, 1)
        actual = (L.dot(e.T) % 2).T.astype(np.uint8)    # (m, 1)
        fails += int(np.sum(pred.ravel() != actual.ravel()))
        done += m
    pL = fails / n_shots
    # Wilson-ish binomial std error
    se = np.sqrt(max(pL * (1 - pL), 1e-12) / n_shots)
    return pL, se


def run():
    sigma = 0.30
    pbar = marginal_pbar(sigma)
    print(f"# sigma = {sigma},  marginal per-qubit flip rate pbar = {pbar:.4f}")
    print(f"# (repetition-code threshold under independent noise = 0.5, so pbar is well below)\n")

    distances = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    # xi values: 0.01 ~ independent; grows to >> d ~ fully shared
    xis = [0.01, 1.0, 3.0, 10.0, 1e4]
    n_shots = 400_000

    results = {}
    for xi in xis:
        row = []
        for d in distances:
            pL, se = sample_logical_error_rate(d, sigma, xi, n_shots)
            row.append((d, pL, se))
        results[xi] = row

    # print table
    print("Logical error rate p_L(d):")
    header = "  d  " + "".join([f"   xi={xi:<8}" for xi in xis])
    print(header)
    for k, d in enumerate(distances):
        line = f" {d:3d} "
        for xi in xis:
            pL = results[xi][k][1]
            line += f"   {pL:.5e}"
        print(line)

    print("\nDistance-suppression factor Lambda(d) = p_L(d)/p_L(d+2):")
    hdr = "  d->d+2 " + "".join([f"  xi={xi:<8}" for xi in xis])
    print(hdr)
    for k in range(len(distances) - 1):
        d = distances[k]
        line = f" {d:2d}->{distances[k+1]:2d} "
        for xi in xis:
            pL0 = results[xi][k][1]
            pL1 = results[xi][k + 1][1]
            lam = pL0 / pL1 if pL1 > 0 else float('inf')
            line += f"   {lam:7.3f}"
        print(line)

    np.save("rep_results.npy", {"distances": distances, "xis": xis,
                                "sigma": sigma, "pbar": pbar, "results": results},
            allow_pickle=True)
    print("\nsaved rep_results.npy")


if __name__ == "__main__":
    run()
