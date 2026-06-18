"""
Multi-GPU d=5 coherent-scaling run (the fixed, batched, checkpointed version).

Splits the coherent + Pauli shots for every theta across N physical GPUs (default
2), one worker process per GPU pinned via CUDA_VISIBLE_DEVICES. Each worker saves
partial sums to its own JSON checkpoint after each <=4 min section, so progress is
durable and resumable. When both workers finish, results are aggregated into the
d=5 coherent/Pauli ratios and written to results/coherent_scaling_d5_multigpu.csv.

This is independent of (and safe to run alongside) any single-GPU run on another
device -- it only touches the GPUs listed in --gpus.

Usage:
  .venv/Scripts/python.exe experiments/run_d5_multigpu.py --gpus 2 3
"""
import os, sys, json, csv, math, time, argparse, subprocess

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
THETAS = [0.20, 0.30, 0.40, 0.50]


def split(total, parts, i):
    return total // parts + (1 if i < total % parts else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--coh-shots", type=int, default=300)       # total coherent shots / theta
    ap.add_argument("--pauli-shots", type=int, default=120000)  # total pauli shots / theta
    ap.add_argument("--max-sec", type=float, default=240.0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    n = len(args.gpus)
    ckpts, procs = [], []
    for i, g in enumerate(args.gpus):
        ck = os.path.join(RESULTS, f"d5_ckpt_gpu{g}.json")
        ckpts.append(ck)
        cmd = [sys.executable, os.path.join(HERE, "_d5_worker.py"),
               "--gpu", str(g), "--ckpt", ck,
               "--coh-shots", str(split(args.coh_shots, n, i)),
               "--pauli-shots", str(split(args.pauli_shots, n, i)),
               "--seed", str(1000 + g), "--max-sec", str(args.max_sec)]
        print("launch:", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd))

    rc = [p.wait() for p in procs]
    print("worker exit codes:", rc, flush=True)

    # ---- aggregate checkpoints across GPUs ----
    fields = ("coh_sum", "coh_sumsq", "coh_n", "pauli_fails", "pauli_n")
    agg = {f"{t:.2f}": {f: 0.0 for f in fields} for t in THETAS}
    for ck in ckpts:
        if not os.path.exists(ck):
            continue
        s = json.load(open(ck))
        for k, v in s.items():
            if k == "_meta":
                continue
            for f in fields:
                agg[k][f] += v.get(f, 0)

    rows = []
    print("\n=== d=5 multi-GPU coherent vs Pauli twirl (whole patch) ===")
    print("  (coh +/- = standard error of the soft estimator; ratio +/- = propagated)")
    for t in THETAS:
        a = agg[f"{t:.2f}"]
        nc = a["coh_n"]; npa = a["pauli_n"]
        coh = a["coh_sum"] / nc if nc else float("nan")
        # SE of the soft coherent mean: sqrt(var/n), var = E[x^2]-E[x]^2
        var = max(a["coh_sumsq"] / nc - coh * coh, 0.0) if nc else float("nan")
        se_coh = math.sqrt(var / nc) if nc else float("nan")
        pau = a["pauli_fails"] / npa if npa else float("nan")
        se_pau = math.sqrt(max(pau * (1 - pau), 0.0) / npa) if npa else float("nan")
        ratio = coh / pau if pau else float("nan")
        se_ratio = (ratio * math.sqrt((se_coh / coh) ** 2 + (se_pau / pau) ** 2)
                    if (pau and coh) else float("nan"))
        rows.append(dict(d=5, theta=t, coh_shots=int(nc), pauli_shots=int(npa),
                         coherent_LER=coh, coherent_SE=se_coh,
                         pauli_LER=pau, pauli_SE=se_pau, ratio=ratio, ratio_SE=se_ratio))
        print(f"  d= 5 theta={t:.2f}  coh={coh:.4e}+/-{se_coh:.1e}  "
              f"pauli={pau:.4e}+/-{se_pau:.1e}  ratio={ratio:.3f}+/-{se_ratio:.3f}")

    out = os.path.join(RESULTS, "coherent_scaling_d5_multigpu.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        [w.writerow(r) for r in rows]
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
