"""
Single-GPU worker for the d=5 coherent-scaling run.

Pinned to ONE physical GPU (via CUDA_VISIBLE_DEVICES), it processes an assigned
shard of shots for every theta and checkpoints partial sums to a JSON file after
each chunk. Chunks are time-bounded (default <= 4 min) so that an interruption
loses at most one section's worth of work, and a restart resumes from the saved
checkpoint. Launched by run_d5_multigpu.py; not meant to be run by hand.
"""
import os, sys, json, time, argparse
import numpy as np


def save_atomic(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)            # physical GPU id
    ap.add_argument("--ckpt", required=True)                     # checkpoint json
    ap.add_argument("--coh-shots", type=int, required=True)      # coherent shots / theta (this worker)
    ap.add_argument("--pauli-shots", type=int, required=True)    # pauli shots / theta (this worker)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-sec", type=float, default=240.0)      # target wall-time per saved section
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--distance", type=int, default=5)           # code distance d
    args = ap.parse_args()

    # pin to one physical GPU BEFORE importing cupy/coherent_sim
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["COHSIM_BACKEND"] = "cupy"
    os.environ.setdefault("COHSIM_BATCH_BYTES", "2.4e9")   # B=4 at d=5; in-place peak ~5.4 GB (safe on every card)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from coherent_sim import CoherentSim

    thetas = args.thetas
    rng = np.random.default_rng(args.seed)
    S = CoherentSim(args.distance)
    disk = list(range(S.n))

    state = json.load(open(args.ckpt)) if os.path.exists(args.ckpt) else {}
    state.setdefault("_meta", {"gpu": args.gpu, "done": False})

    rate = None  # coherent shots/sec estimate, refined as we go
    for t in thetas:
        key = f"{t:.2f}"
        st = state.setdefault(key, {"coh_sum": 0.0, "coh_sumsq": 0.0, "coh_n": 0,
                                    "pauli_fails": 0, "pauli_n": 0})

        # ---- coherent (GPU, batched), time-bounded chunks ----
        while st["coh_n"] < args.coh_shots:
            remaining = args.coh_shots - st["coh_n"]
            if rate is None:
                chunk = min(remaining, max(S.B, S.B * 4))        # small calibration chunk
            else:
                chunk = min(remaining, max(S.B, int(rate * args.max_sec)))
            t0 = time.time()
            s, ss, _ = S.coherent_stats(disk, t, t, chunk, rng)
            dt = max(time.time() - t0, 1e-6)
            st["coh_sum"] += s
            st["coh_sumsq"] += ss
            st["coh_n"] += chunk
            save_atomic(args.ckpt, state)
            rate = chunk / dt
            print(f"[gpu{args.gpu}] theta={key} coherent {st['coh_n']}/{args.coh_shots} "
                  f"(+{chunk} in {dt:.1f}s, ~{rate:.2f} sh/s)", flush=True)

        # ---- pauli twirl (vectorized CPU), chunked + checkpointed ----
        while st["pauli_n"] < args.pauli_shots:
            remaining = args.pauli_shots - st["pauli_n"]
            chunk = min(remaining, 20000)
            m = S.pauli_twirl_LER(disk, t, t, chunk, rng)
            st["pauli_fails"] += int(round(float(m) * chunk))
            st["pauli_n"] += chunk
            save_atomic(args.ckpt, state)
            print(f"[gpu{args.gpu}] theta={key} pauli {st['pauli_n']}/{args.pauli_shots}", flush=True)

    state["_meta"]["done"] = True
    save_atomic(args.ckpt, state)
    print(f"[gpu{args.gpu}] DONE", flush=True)


if __name__ == "__main__":
    main()
