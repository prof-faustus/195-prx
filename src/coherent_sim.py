"""
coherent_sim.py  --  GPU-ready coherent-burst simulator for surface-code patches.

State vector stored as a length-2^n complex array; single-qubit gates applied by
tensor reshaping (NOT 2^n x 2^n matrices), so it scales to d=5 (25 qubits, ~0.5 GB
complex128) and d=7 via tensor-network backends.  Backend is swappable:

    import numpy as xp                      # CPU (this file's default)
    # import cupy as xp                      # GPU: set BACKEND='cupy'

Set environment variable COHSIM_BACKEND=cupy to run on NVIDIA GPUs unchanged.

Scientific purpose: measure the logical error rate of a coherent (non-Pauli,
optionally non-commuting) burst on a real rotated surface code, and compare it to
the Pauli twirl of the same burst, as a function of code distance d.  The open
question (Route C) is whether the coherent excess over the Pauli model persists or
grows with d -- which would defeat distance suppression -- or decays.
"""
import os
import numpy as np
import stim
import pymatching
import scipy.sparse as sp

_BK = os.environ.get("COHSIM_BACKEND", "numpy")
if _BK == "cupy":
    import cupy as xp           # noqa
else:
    import numpy as xp


# ---------------------------------------------------------------- code extraction
def code_data(d):
    """Return (datax, dataz, coords, HxX, LX, HzZ, LZ) for distance-d rotated code.
    HxX: X-stabilizers (detect Z errors), LX: logical-X support (over data, datax order).
    HzZ: Z-stabilizers (detect X errors), LZ: logical-Z support."""
    cx = stim.Circuit.generated("surface_code:rotated_memory_x", rounds=1, distance=d)
    cz = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=1, distance=d)
    datax = [t.value for t in next(i for i in cx if i.name == "RX").targets_copy()]
    cmx, cmz = {}, {}
    for inst in cx:
        if inst.name == "QUBIT_COORDS":
            cmx[inst.targets_copy()[0].value] = tuple(inst.gate_args_copy())
    for inst in cz:
        if inst.name == "QUBIT_COORDS":
            cmz[inst.targets_copy()[0].value] = tuple(inst.gate_args_copy())
    c2z = {v: k for k, v in cmz.items()}
    dataz = [c2z[cmx[q]] for q in datax]
    coords = np.array([cmx[q] for q in datax], float)

    def supports(circ, gate, data):
        base = list(circ); ndet = circ.num_detectors
        H = np.zeros((ndet, len(data)), np.uint8); L = np.zeros((1, len(data)), np.uint8)
        for j, q in enumerate(data):
            new = stim.Circuit(); placed = False
            for inst in base:
                new.append(inst)
                if inst.name == "R" and not placed:
                    new.append(gate, [q], 0.01); placed = True
            for e in new.detector_error_model(decompose_errors=False, flatten_loops=True):
                if e.type == "error":
                    for t in e.targets_copy():
                        if t.is_relative_detector_id(): H[t.val, j] = 1
                        if t.is_logical_observable_id(): L[t.val, j] = 1
        return H[H.sum(1) > 0], L

    HxX, LX = supports(cx, "Z_ERROR", datax)
    HzZ, LZ = supports(cz, "X_ERROR", dataz)
    return datax, dataz, coords, HxX, LX, HzZ, LZ


# ---------------------------------------------------------------- state-vector ops
def apply_1q(psi, g, i, n):
    """Apply 2x2 gate g to qubit i of state psi (shape (2^n,)). Reshape-based."""
    psi = psi.reshape((2 ** i, 2, 2 ** (n - i - 1)))
    psi = xp.tensordot(g, psi, axes=([1], [1]))      # contract gate's col index
    psi = xp.moveaxis(psi, 0, 1)
    return psi.reshape(2 ** n)


def apply_pauli_string(psi, paulis, support, n):
    """Apply a product of single-qubit Paulis (X or Z 2x2 matrices) on support qubits."""
    for i in support:
        psi = apply_1q(psi, paulis, i, n)
    return psi


# ----------------------------------------------- batched state-vector ops (B trajectories)
# All ops below mutate the batched state IN PLACE (no per-op 2^n allocations) and reuse
# caller-provided scratch, so the d=5 working set stays ~2x the state instead of ~7x.
def _host(a):
    """Move an xp array (numpy or cupy) to a host numpy array."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _slabs(psi, i, n):
    """Return the (|0>, |1>) half-slab views of qubit i over a batch psi (B, 2^n)."""
    B = psi.shape[0]
    v = psi.reshape((B, 2 ** i, 2, 2 ** (n - i - 1)))
    return v[:, :, 0, :], v[:, :, 1, :]


def apply_g_inplace(psi, g4, i, n, scr):
    """Apply a general 2x2 gate g4=(g00,g01,g10,g11) to qubit i in place; scr is a half-state buffer."""
    s0, s1 = _slabs(psi, i, n)
    g00, g01, g10, g11 = g4
    t = scr[:s0.size].reshape(s0.shape)
    t[:] = s0                       # save old |0> slab
    s0 *= g00; s0 += g01 * s1       # s0 <- g00*s0 + g01*s1
    s1 *= g11; s1 += g10 * t        # s1 <- g10*s0_old + g11*s1
    return psi


def apply_x_inplace(psi, i, n, scr):
    """Apply X to qubit i in place (swap |0>/|1> slabs); scr is a half-state buffer."""
    s0, s1 = _slabs(psi, i, n)
    t = scr[:s0.size].reshape(s0.shape)
    t[:] = s0; s0[:] = s1; s1[:] = t
    return psi


def apply_z_inplace(psi, i, n):
    """Apply Z to qubit i in place (negate the |1> slab)."""
    _, s1 = _slabs(psi, i, n)
    s1 *= -1
    return psi


def _real_braket(psi, phi):
    """Re<psi|phi> per trajectory, (B,), without allocating a conjugate of the full state."""
    return (psi.real * phi.real).sum(axis=1) + (psi.imag * phi.imag).sum(axis=1)


def measure_collapse_inplace(psi, Spsi, kind, supp, n, scr, rng):
    """Project the batch onto a +/-1 eigenspace of stabilizer `kind`(=X/Z) on supp, per trajectory.
    `Spsi` is a reused full-state scratch buffer; psi is collapsed in place. Returns host bits (B,)."""
    Spsi[:] = psi                                       # one reuse-buffer copy per measurement
    if kind == "X":
        for q in supp:
            apply_x_inplace(Spsi, q, n, scr)
    else:
        for q in supp:
            apply_z_inplace(Spsi, q, n)
    e = _host(_real_braket(psi, Spsi))                  # <psi|S|psi> per trajectory
    prob_plus = (e + 1.0) / 2.0
    plus = rng.random(prob_plus.shape[0]) < prob_plus
    sign = xp.asarray(np.where(plus, 1.0, -1.0))
    Spsi *= sign[:, None]                               # +/- per trajectory, in place
    psi += Spsi                                         # psi <- psi +/- S psi
    nrm = xp.sqrt((psi.real ** 2).sum(axis=1) + (psi.imag ** 2).sum(axis=1))
    psi /= nrm[:, None]
    return (~plus).astype(np.uint8)


def apply_correction_b(psi, gate_is_x, qubit, n, mask):
    """Apply X (gate_is_x=True) or Z to `qubit` only on the masked trajectories (mask: xp bool (B,))."""
    if not bool(mask.any()):
        return psi
    B = psi.shape[0]
    v = psi.reshape((B, 2 ** qubit, 2, 2 ** (n - qubit - 1)))
    if gate_is_x:
        v[mask] = v[mask][:, :, ::-1, :]                 # X swaps the |0>/|1> slabs
    else:
        v[mask, :, 1, :] = -v[mask, :, 1, :]             # Z flips sign of |1> slab
    return psi


class CoherentSim:
    Xg = xp.array([[0, 1], [1, 0]], complex)
    Zg = xp.array([[1, 0], [0, -1]], complex)

    def __init__(self, d):
        self.d = d
        (self.datax, self.dataz, self.coords,
         self.HxX, self.LX, self.HzZ, self.LZ) = code_data(d)
        self.n = len(self.datax)
        self.xsupp = [np.where(r)[0] for r in self.HxX]      # X-stab supports
        self.zsupp = [np.where(r)[0] for r in self.HzZ]      # Z-stab supports
        self.logZ = np.where(self.LZ[0])[0]
        # decoders returning physical correction (faults = identity)
        self.McX = pymatching.Matching(self.HzZ, faults_matrix=sp.identity(self.n, format="csc"))
        self.McZ = pymatching.Matching(self.HxX, faults_matrix=sp.identity(self.n, format="csc"))
        self.MlX = pymatching.Matching(self.HzZ, faults_matrix=self.LZ)
        self.MlZ = pymatching.Matching(self.HxX, faults_matrix=self.LX)
        self.psi0 = self._prep_logical_zero()
        # batch size: how many trajectories share one GPU state-vector pass.
        # keep resident batch under COHSIM_BATCH_BYTES (default 2 GB); 1q ops need ~3x transiently.
        budget = float(os.environ.get("COHSIM_BATCH_BYTES", 2.0e9))
        self.B = max(1, min(512, int(budget // ((2 ** self.n) * 16))))
        self.HzZ_i = self.HzZ.astype(np.int64)

    def _prep_logical_zero(self):
        n = self.n
        psi = xp.zeros(2 ** n, complex); psi[0] = 1.0
        for supp in self.xsupp:                              # project onto +1 of each X-stab
            Spsi = apply_pauli_string(psi.copy(), self.Xg, supp, n)
            psi = (psi + Spsi) / 2
        return psi / xp.linalg.norm(psi)

    def _expect_ZL(self, psi):
        Zpsi = apply_pauli_string(psi.copy(), self.Zg, self.logZ, self.n)
        return float(xp.real(xp.vdot(psi, Zpsi)))

    def _measure_collapse(self, psi, paulis, supp, rng):
        Spsi = apply_pauli_string(psi.copy(), paulis, supp, self.n)
        prob_plus = float(xp.real(xp.vdot(psi, Spsi)) + 1.0) / 2.0
        if rng.random() < prob_plus:
            out = (psi + Spsi); return out / xp.linalg.norm(out), 0
        out = (psi - Spsi); return out / xp.linalg.norm(out), 1

    def _coherent_batch(self, g4, disk, b, rng):
        """Run b coherent trajectories at once (in place); return soft logical-error contributions (b,)."""
        n = self.n
        psi = xp.tile(self.psi0, (b, 1))                  # (b, 2^n), fresh copy per batch
        scr = xp.empty(b * 2 ** (n - 1), complex)         # one reused half-state scratch
        Spsi = xp.empty((b, 2 ** n), complex)             # one reused full-state scratch
        for i in disk:
            apply_g_inplace(psi, g4, i, n, scr)
        synX = np.zeros((b, len(self.zsupp)), np.uint8)   # Z-stabs detect X
        for k, supp in enumerate(self.zsupp):
            synX[:, k] = measure_collapse_inplace(psi, Spsi, "Z", supp, n, scr, rng)
        synZ = np.zeros((b, len(self.xsupp)), np.uint8)   # X-stabs detect Z
        for k, supp in enumerate(self.xsupp):
            synZ[:, k] = measure_collapse_inplace(psi, Spsi, "X", supp, n, scr, rng)
        corrX = self.McX.decode_batch(synX)               # (b, n) X-corrections
        corrZ = self.McZ.decode_batch(synZ)               # (b, n) Z-corrections
        for i in range(n):
            apply_correction_b(psi, True, i, n, xp.asarray(corrX[:, i].astype(bool)))
        for i in range(n):
            apply_correction_b(psi, False, i, n, xp.asarray(corrZ[:, i].astype(bool)))
        Spsi[:] = psi
        for q in self.logZ:
            apply_z_inplace(Spsi, q, n)
        zl = _host(_real_braket(psi, Spsi))               # <Z_L> per trajectory
        return (1.0 - zl) / 2.0

    def _burst_gate(self, tx, tz):
        """exp(-i phi (nx X + nz Z)) as the python-complex tuple (g00,g01,g10,g11)."""
        phi = float(np.hypot(tx, tz)); nx, nz = tx / phi, tz / phi
        c, s = float(np.cos(phi)), float(np.sin(phi))
        return (complex(c, -s * nz), complex(0.0, -s * nx),
                complex(0.0, -s * nx), complex(c, s * nz))

    def coherent_stats(self, disk, tx, tz, shots, rng):
        """Return (sum, sumsq, n) of the per-shot soft logical-error contributions.
        Sum/sumsq let callers form the mean and its standard error across shards."""
        g4 = self._burst_gate(tx, tz)
        s = 0.0; ss = 0.0; done = 0
        while done < shots:
            b = min(self.B, shots - done)
            c = self._coherent_batch(g4, disk, b, rng)
            s += float(c.sum()); ss += float((c * c).sum()); done += b
        return s, ss, shots

    def coherent_LER(self, disk, tx, tz, shots, rng):
        """Logical error rate of a coherent burst exp(-i sum (tx X_i + tz Z_i)) on `disk`.
        Trajectories are processed in batches of self.B on the active backend (GPU under cupy)."""
        s, _, n = self.coherent_stats(disk, tx, tz, shots, rng)
        return s / n

    def pauli_twirl_LER(self, disk, tx, tz, shots, rng):
        """Logical error rate of the Pauli twirl of the same burst (fully incoherent), vectorized."""
        phi = float(np.hypot(tx, tz)); nx, nz = tx / phi, tz / phi
        pX = np.sin(phi) ** 2 * nx * nx
        disk = np.array(disk); n = self.n
        eX = np.zeros((shots, n), np.uint8)
        eX[:, disk] = (rng.random((shots, len(disk))) < pX).astype(np.uint8)
        syn = ((self.HzZ_i @ eX.T) % 2).T.astype(np.uint8)        # (shots, n_Zstab)
        corr = self.McX.decode_batch(syn)                          # (shots, n)
        res = (eX ^ corr) % 2
        fails = ((self.LZ @ res.T) % 2)[0]                         # logical-Z parity per shot
        return int(fails.sum()) / shots


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    S = CoherentSim(3)
    print(f"backend={_BK}  n={S.n}  <Z_L>|0_L = {S._expect_ZL(S.psi0):.6f} (want +1)")
    allq = list(range(S.n))
    print("\nValidation vs dense d=3 (coherent should be >Pauli, ratio ~1.4-2.3):")
    print(" t      coherent     pauli       ratio")
    for t in [0.25, 0.35, 0.45, 0.55]:
        c = S.coherent_LER(allq, t, t, 3000, rng)
        p = S.pauli_twirl_LER(allq, t, t, 80000, rng)
        print(f" {t:.2f}   {c:.4e}   {p:.4e}   {c/p:.3f}")
