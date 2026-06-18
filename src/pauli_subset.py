"""
Subset-sampling (Bravyi-Vargo style) estimator for the Pauli-twirl logical error
rate of a surface-code patch -- the estimator that stays defined where uniform
Monte-Carlo returns zero failures (the low-theta tail of the d=5 sweep).

Idea: write the twirl LER as a sum over error weight w,
    LER = sum_w  Binom(w; n, p) * f_w,
where f_w is the conditional logical-failure fraction among weight-w X-error
patterns (uniform within the weight class), estimated by sampling (or enumerating,
for small classes) patterns of *exactly* weight w and decoding each with MWPM.
Because each f_w is measured inside its weight class, the estimate has small
relative variance even when the overall LER is ~1e-8 and uniform MC sees nothing.

Per-qubit X probability for the t_x=t_z=theta burst twirl (matching
CoherentSim.pauli_twirl_LER):  p = sin^2(phi) * n_x^2,  phi=hypot(theta,theta),
n_x = theta/phi = 1/sqrt(2)  =>  p = 0.5 * sin^2(sqrt(2) * theta).

This module is pure numpy + stim + pymatching (no state vector, no cupy): it builds
only the Z-stabilizer check matrix, the logical-Z support, and the MWPM decoder.
"""
import os, sys, math
from math import comb
from itertools import combinations
import numpy as np
import scipy.sparse as sp
import pymatching

sys.path.insert(0, os.path.dirname(__file__))
from coherent_sim import code_data


def pX_of_theta(theta):
    """Per-qubit X-error probability of the twirl of the t_x=t_z=theta burst."""
    phi = math.hypot(theta, theta)
    nx = theta / phi
    return math.sin(phi) ** 2 * nx * nx


class PauliDecoder:
    """Lightweight Pauli-twirl LER machinery for distance d (no state vector)."""

    def __init__(self, d):
        _, _, _, _, _, HzZ, LZ = code_data(d)
        self.n = HzZ.shape[1]
        self.HzZ = HzZ.astype(np.int64)
        self.LZ = LZ.astype(np.int64)
        self.McX = pymatching.Matching(HzZ, faults_matrix=sp.identity(self.n, format="csc"))

    def _fail(self, eX):
        """Logical-Z failure (0/1) per row of eX (shape (N, n) uint8), MWPM-decoded."""
        syn = ((self.HzZ @ eX.T) % 2).T.astype(np.uint8)
        corr = self.McX.decode_batch(syn)
        res = (eX ^ corr) % 2
        return ((self.LZ @ res.T) % 2)[0]

    def uniform_LER(self, disk, p, shots, rng):
        """Plain uniform Monte-Carlo twirl LER (the reference to validate against)."""
        disk = np.array(disk)
        eX = np.zeros((shots, self.n), np.uint8)
        eX[:, disk] = (rng.random((shots, len(disk))) < p).astype(np.uint8)
        f = self._fail(eX)
        ler = f.mean()
        se = math.sqrt(max(ler * (1 - ler), 0.0) / shots)
        return ler, se, int(f.sum())

    def subset_LER(self, disk, p, shots_per_w, rng, enum_cap=20000, pmf_floor=1e-16):
        """Subset-sampling twirl LER. Returns (LER, SE, details).
        For each weight w with non-negligible Binom(w;nd,p): enumerate all C(nd,w)
        weight-w patterns if that count <= enum_cap (exact f_w), else sample
        shots_per_w of them. LER = sum_w pmf_w * f_w; variance propagated from the
        sampled weight classes (enumerated classes are exact)."""
        disk = np.array(disk); nd = len(disk)
        # binomial pmf over weights via log-space (stable for small p, large nd)
        ler = 0.0; var = 0.0; details = []
        logp = math.log(p) if p > 0 else -math.inf
        log1mp = math.log1p(-p)
        for w in range(1, nd + 1):
            logpmf = (math.lgamma(nd + 1) - math.lgamma(w + 1) - math.lgamma(nd - w + 1)
                      + w * logp + (nd - w) * log1mp)
            pmf = math.exp(logpmf) if logpmf > -700 else 0.0
            if pmf < pmf_floor:
                # past the peak with negligible mass -> stop; below the peak just skip
                if w > nd * p:
                    break
                continue
            ntot = comb(nd, w)
            if ntot <= enum_cap:
                idx = np.array(list(combinations(range(nd), w)), dtype=np.int64)
                eX = np.zeros((idx.shape[0], self.n), np.uint8)
                rows = np.repeat(np.arange(idx.shape[0]), w)
                eX[rows, disk[idx.ravel()]] = 1
                f = self._fail(eX)
                fw = float(f.mean()); N = idx.shape[0]; exact = True
            else:
                N = shots_per_w
                eX = np.zeros((N, self.n), np.uint8)
                for i in range(N):
                    sel = rng.choice(nd, size=w, replace=False)
                    eX[i, disk[sel]] = 1
                f = self._fail(eX)
                fw = float(f.mean()); exact = False
            ler += pmf * fw
            if not exact:
                var += pmf * pmf * (fw * (1 - fw) / N)
            details.append((w, pmf, fw, N, exact))
        return ler, math.sqrt(var), details


def validate(d=5, thetas=(0.12, 0.14, 0.16, 0.18, 0.20), uniform_shots=2_000_000,
             shots_per_w=4000, seed=0):
    """Validate subset vs uniform MC in the overlap region (must agree within bars)."""
    rng = np.random.default_rng(seed)
    P = PauliDecoder(d)
    disk = list(range(P.n))
    print(f"d={d}  n={P.n}  validate subset vs uniform MC ({uniform_shots} shots)")
    print(" theta     p       uniform_LER +/- SE        subset_LER +/- SE        n_sigma")
    ok = True
    for th in thetas:
        p = pX_of_theta(th)
        u, ue, nf = P.uniform_LER(disk, p, uniform_shots, rng)
        s, se, det = P.subset_LER(disk, p, shots_per_w, rng)
        comb_se = math.sqrt(ue * ue + se * se) or 1e-30
        nsig = abs(u - s) / comb_se
        ok = ok and (nsig < 3 or abs(u - s) < 3e-6)
        print(f" {th:.2f}  {p:.4e}  {u:.4e} +/- {ue:.1e} ({nf:6d} fails)   "
              f"{s:.4e} +/- {se:.1e}   {nsig:4.1f}")
    print("OVERLAP AGREEMENT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    validate()
