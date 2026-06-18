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

    def coherent_LER(self, disk, tx, tz, shots, rng):
        """Logical error rate of a coherent burst exp(-i sum (tx X_i + tz Z_i)) on `disk`."""
        phi = float(np.hypot(tx, tz)); nx, nz = tx / phi, tz / phi
        g = float(np.cos(phi)) * xp.eye(2) - 1j * float(np.sin(phi)) * (nx * self.Xg + nz * self.Zg)
        tot = 0.0
        for _ in range(shots):
            psi = self.psi0.copy()
            for i in disk:
                psi = apply_1q(psi, g, i, self.n)
            # full syndrome
            synX = np.zeros(len(self.zsupp), np.uint8)        # Z-stabs detect X
            for k, supp in enumerate(self.zsupp):
                psi, b = self._measure_collapse(psi, self.Zg, supp, rng); synX[k] = b
            synZ = np.zeros(len(self.xsupp), np.uint8)        # X-stabs detect Z
            for k, supp in enumerate(self.xsupp):
                psi, b = self._measure_collapse(psi, self.Xg, supp, rng); synZ[k] = b
            for i in np.where(self.McX.decode(synX))[0]:
                psi = apply_1q(psi, self.Xg, i, self.n)
            for i in np.where(self.McZ.decode(synZ))[0]:
                psi = apply_1q(psi, self.Zg, i, self.n)
            tot += (1.0 - self._expect_ZL(psi)) / 2.0
        return tot / shots

    def pauli_twirl_LER(self, disk, tx, tz, shots, rng):
        """Logical error rate of the Pauli twirl of the same burst (fully incoherent)."""
        phi = float(np.hypot(tx, tz)); nx, nz = tx / phi, tz / phi
        pX = np.sin(phi) ** 2 * nx * nx; pZ = np.sin(phi) ** 2 * nz * nz
        disk = np.array(disk); n = self.n; fails = 0
        for _ in range(shots):
            eX = np.zeros(n, np.uint8); eZ = np.zeros(n, np.uint8)
            eX[disk[rng.random(len(disk)) < pX]] = 1
            eZ[disk[rng.random(len(disk)) < pZ]] = 1
            cX = self.McX.decode(((self.HzZ @ eX) % 2).astype(np.uint8))
            resX = (eX ^ cX) % 2
            fails += int((self.LZ @ resX)[0] % 2)
        return fails / shots


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
