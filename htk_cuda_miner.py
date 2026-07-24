#!/usr/bin/env python3
"""
htk_cuda_miner — multi-GPU CUDA miner for HashToken (HTK), built to run on
vast.ai rigs and feed the existing hashtoken-ntfy pipeline.

What it does
------------
- Compiles the UNMODIFIED `keccak_miner.cu` at runtime (CuPy RawModule / nvcc)
  and runs `find_solution_kernel` — `keccak256(nonce ‖ prev_hash) <= max_value`,
  the exact HTK contract rule that ../listener.py re-verifies before minting.
- One OS process per GPU (no GIL contention; a CUDA fault on one GPU can't take
  the others down — the worker is auto-restarted).
- Provably NON-OVERLAPPING nonce ranges across every GPU on every rig: each GPU
  is a global worker `W` of `N` total, and uses base_nonce_part = W + k*N for
  k = 0,1,2,...  Because base_nonce_part occupies nonce bytes 0-7 and thread_id
  occupies bytes 8-15 (disjoint byte ranges), distinct base_nonce_part values
  always yield disjoint nonce sets — so a strided partition of base_nonce_part
  tiles the whole space with zero overlap and zero gaps.
- Per-GPU auto-tuning: sweeps block sizes, then GROWS the grid until each launch
  lasts ~--target-launch-seconds. Long GPU launches keep the CUDA cores busy and
  the host nearly idle (few round-trips/sec), as requested.
- Auto-fetches prev_hash/max_value on-chain and auto-restarts the search when
  prev_hash rotates.
- On a hit: pushes the 64-hex nonce to the ntfy NONCE topic (what listener.py
  consumes) and appends it to a local JSONL so nothing is ever lost.
- Startup banner + error/warning alerts go to a SEPARATE ntfy STATUS topic so the
  nonce channel stays clean. No heartbeat (hashrate is logged to stdout only).

See README.md for vast.ai setup and the multi-rig launch formula.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CU_PATH = SCRIPT_DIR / "keccak_miner.cu"
KERNEL_NAME = "find_solution_kernel"
MASK64 = (1 << 64) - 1

# ── HTK chain constants (identical to ../listener.py) ───────────────────────────
CONTRACT = "0xE5544a2A5fA9b175da60D8Eec67adD5582bB31b0"
SEL_PREV_HASH = "0xc69b5df2"   # prev_hash() -> bytes32
SEL_MAX_VALUE = "0x98597629"   # max_value() -> uint256
DEFAULT_READ_RPC = os.environ.get("HTK_READ_RPC", "https://ethereum-rpc.publicnode.com")

# ── ntfy defaults (nonce topic matches ../listener.py) ──────────────────────────
DEFAULT_NTFY_BASE = os.environ.get("NTFY_BASE", "https://ntfy.sh").rstrip("/")
DEFAULT_NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "htk-nonce-a303558a1aa9e6043d43531d")
DEFAULT_STATUS_TOPIC = os.environ.get("NTFY_STATUS_TOPIC", "")  # empty => local only

TUNE_CACHE = SCRIPT_DIR / ".tune_cache.json"


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(*a):
    print(now(), *a, flush=True)


# ════════════════════════════════════════════════════════════════════════════════
# GPU worker (runs in its own process)
# ════════════════════════════════════════════════════════════════════════════════

def _compile_kernel(cu_source: str):
    """Compile keccak_miner.cu and return the callable RawKernel.

    Primary path: backend='nvcc' (real compiler, handles <cuda_runtime.h>).
    Fallback: backend='nvrtc' with the cuda_runtime.h include stripped in-memory
    (the on-disk .cu is never modified; hashing logic is untouched).
    """
    import cupy as cp

    try:
        mod = cp.RawModule(
            code=cu_source,
            backend="nvcc",
            options=("-O3", "-std=c++14"),
            name_expressions=[KERNEL_NAME],
        )
        return mod.get_function(KERNEL_NAME)
    except Exception as e_nvcc:
        stripped = "\n".join(
            ln for ln in cu_source.splitlines()
            if "cuda_runtime.h" not in ln
        )
        try:
            mod = cp.RawModule(
                code=stripped,
                backend="nvrtc",
                options=("-std=c++14",),
                name_expressions=[KERNEL_NAME],
            )
            return mod.get_function(KERNEL_NAME)
        except Exception as e_nvrtc:
            raise RuntimeError(
                f"kernel compile failed (nvcc: {e_nvcc}; nvrtc: {e_nvrtc})"
            )


def _benchmark(cp, kern, d_prev, d_target_zero, d_nonce, d_flag, block, grid):
    """Return hashes/sec for (block, grid) using an impossible (all-zero) target."""
    import numpy as np
    d_flag.fill(0)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    kern((grid,), (block,), (d_prev, d_target_zero, d_nonce, np.uint64(0), d_flag))
    end.record()
    end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end)
    threads = block * grid
    return threads / (ms / 1000.0) if ms > 0 else 0.0


def _autotune(cp, kern, dev, d_prev, d_nonce, d_flag, block_sizes, target_secs):
    """Pick the best block size, then scale grid so a launch lasts ~target_secs."""
    import numpy as np
    d_target_zero = cp.zeros(32, dtype=cp.uint8)  # all-zero target => never a hit
    attrs = dev.attributes
    sm = attrs.get("MultiProcessorCount", 16)
    max_grid = attrs.get("MaxGridDimX", 2**31 - 1)

    best = None  # (hashrate, block)
    calib_grid_factor = 64  # blocks/SM for the short calibration launch
    for block in block_sizes:
        grid = max(1, sm * calib_grid_factor)
        # warm-up (JIT/first-launch cost excluded from the measurement)
        _benchmark(cp, kern, d_prev, d_target_zero, d_nonce, d_flag, block, grid)
        hr = _benchmark(cp, kern, d_prev, d_target_zero, d_nonce, d_flag, block, grid)
        if best is None or hr > best[0]:
            best = (hr, block)
    hashrate, block = best

    # grid needed for a launch of ~target_secs at the measured hashrate
    target_threads = int(hashrate * target_secs)
    grid = max(sm * calib_grid_factor, target_threads // block)
    grid = min(grid, int(max_grid))
    grid = max(grid, 1)
    return block, grid, hashrate


def gpu_worker(gpu_id, worker_id, num_workers, cu_source, args,
               job_buf, job_gen, job_lock, out_q, stop_evt):
    """Mine on one GPU forever (until stop_evt). Reports via out_q."""
    tag = f"gpu{gpu_id}/W{worker_id}"
    try:
        import cupy as cp
        import numpy as np

        dev = cp.cuda.Device(gpu_id)
        dev.use()
        name = cp.cuda.runtime.getDeviceProperties(gpu_id)["name"].decode()

        kern = _compile_kernel(cu_source)

        # Persistent device buffers
        d_prev = cp.zeros(32, dtype=cp.uint8)
        d_target = cp.full(32, 255, dtype=cp.uint8)  # default: never matches
        d_nonce = cp.zeros(32, dtype=cp.uint8)
        d_flag = cp.zeros(1, dtype=cp.int32)

        # Auto-tune (or load cache keyed by GPU name)
        cache = {}
        if TUNE_CACHE.exists():
            try:
                cache = json.loads(TUNE_CACHE.read_text())
            except Exception:
                cache = {}
        ck = f"{name}|{args.target_launch_seconds}"
        if not args.retune and ck in cache:
            block, grid = cache[ck]["block"], cache[ck]["grid"]
            hashrate = cache[ck].get("hashrate", 0.0)
        else:
            block, grid, hashrate = _autotune(
                cp, kern, dev, d_prev, d_nonce, d_flag,
                args.block_sizes, args.target_launch_seconds,
            )
            cache[ck] = {"block": block, "grid": grid, "hashrate": hashrate}
            try:
                TUNE_CACHE.write_text(json.dumps(cache, indent=2))
            except Exception:
                pass

        out_q.put({
            "type": "ready", "worker": worker_id, "gpu": gpu_id, "name": name,
            "block": block, "grid": grid, "hashrate": hashrate,
        })
        log(f"[{tag}] {name}: block={block} grid={grid} "
            f"~{hashrate/1e6:.1f} MH/s per-launch~{block*grid/max(hashrate,1):.2f}s")

        k = 0                 # private strided launch counter (never reset)
        cur_gen = -1          # job generation currently loaded on the device
        prev = b"\x00" * 32   # current prev_hash (for reporting found nonces)
        have_job = False
        stat_hashes = 0
        stat_t0 = time.time()

        while not stop_evt.is_set():
            # Pick up a new job if the coordinator bumped the generation.
            with job_lock:
                gen = job_gen.value
                if gen != cur_gen:
                    prev = bytes(job_buf[0:32])
                    tgt = bytes(job_buf[32:64])
            if gen != cur_gen:
                if any(prev):
                    d_prev.set(np.frombuffer(prev, dtype=np.uint8).copy())
                    d_target.set(np.frombuffer(tgt, dtype=np.uint8).copy())
                    have_job = True
                cur_gen = gen
            if not have_job:
                time.sleep(0.2)
                continue

            base = (worker_id + k * num_workers) & MASK64
            k += 1
            d_flag.fill(0)
            kern((grid,), (block,), (d_prev, d_target, d_nonce, np.uint64(base), d_flag))
            found = int(d_flag.get()[0])
            stat_hashes += block * grid

            if found:
                nonce = bytes(cp.asnumpy(d_nonce))
                out_q.put({
                    "type": "found", "worker": worker_id, "gpu": gpu_id,
                    "nonce": nonce.hex(), "prev_hash": prev.hex(),
                })
                d_nonce.fill(0)

            # periodic local stat (no ntfy) ~ every 5s
            if time.time() - stat_t0 >= 5.0:
                dt = time.time() - stat_t0
                out_q.put({
                    "type": "stat", "worker": worker_id, "gpu": gpu_id,
                    "hashes": stat_hashes, "dt": dt, "name": name,
                })
                stat_hashes = 0
                stat_t0 = time.time()

    except Exception as e:
        try:
            out_q.put({"type": "error", "worker": worker_id, "gpu": gpu_id,
                       "msg": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
        # Non-zero exit so the coordinator respawns us.
        os._exit(1)


# ════════════════════════════════════════════════════════════════════════════════
# Coordinator (parent process)
# ════════════════════════════════════════════════════════════════════════════════

def ntfy_push(base, topic, body, title=None, tags=None, tries=4):
    """POST a message to ntfy with retry/backoff. Returns True on success."""
    import requests
    if not topic:
        return False
    url = f"{base}/{topic}"
    headers = {}
    if title:
        headers["Title"] = title
    if tags:
        headers["Tags"] = tags
    delay = 1.0
    for _ in range(tries):
        try:
            r = requests.post(url, data=body.encode(), headers=headers, timeout=15)
            r.raise_for_status()
            return True
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 15)
    return False


def _eth_call(rpc, to, data):
    """One raw JSON-RPC eth_call; returns the 32-byte result. No web3 needed."""
    import requests
    r = requests.post(
        rpc,
        json={"jsonrpc": "2.0", "id": 1, "method": "eth_call",
              "params": [{"to": to, "data": data}, "latest"]},
        timeout=20,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"eth_call error: {body['error']}")
    result = body.get("result", "0x")
    return bytes.fromhex(result[2:] if result.startswith("0x") else result)


def read_state(rpc):
    """Return (prev_hash_bytes32, max_value_int) via raw JSON-RPC."""
    prev = _eth_call(rpc, CONTRACT, SEL_PREV_HASH)
    mx = _eth_call(rpc, CONTRACT, SEL_MAX_VALUE)
    if len(prev) != 32 or len(mx) != 32:
        raise ValueError("unexpected return length from HTK reads")
    return prev, int.from_bytes(mx, "big")


def chain_poller(args, job_buf, job_gen, job_lock, out_q, stop_evt):
    """Poll HTK; on prev_hash change bump the shared job generation."""
    last_prev = None
    backoff = 1.0
    while not stop_evt.is_set():
        try:
            prev, mx = read_state(args.read_rpc)
            backoff = 1.0
            if prev != last_prev:
                with job_lock:
                    job_buf[0:32] = prev
                    job_buf[32:64] = mx.to_bytes(32, "big")
                    job_gen.value += 1
                last_prev = prev
                log(f"[job] prev_hash=0x{prev.hex()[:16]}… "
                    f"max_value=0x{mx:064x}"[:60] + " (search restarted)")
        except Exception as e:
            out_q.put({"type": "error", "worker": -1, "gpu": -1,
                       "msg": f"RPC read failed: {type(e).__name__}: {e}"})
            time.sleep(backoff)
            backoff = min(backoff * 2, args.poll_interval)
        stop_evt.wait(args.poll_interval)


def spawn_worker(ctx, gpu_id, worker_id, args, cu_source,
                 job_buf, job_gen, job_lock, out_q, stop_evt):
    p = ctx.Process(
        target=gpu_worker,
        args=(gpu_id, worker_id, args.num_workers, cu_source, args,
              job_buf, job_gen, job_lock, out_q, stop_evt),
        daemon=False,
    )
    p.start()
    return p


def run_miner(args, cu_source):
    ctx = mp.get_context("spawn")
    job_buf = ctx.Array("c", 64, lock=False)   # 32 prev_hash + 32 target
    job_lock = ctx.Lock()
    job_gen = ctx.Value("l", 0, lock=False)
    out_q = ctx.Queue()
    stop_evt = ctx.Event()

    # workers: local index -> gpu id ; global worker id = worker_offset + local
    procs = {}   # worker_id -> (gpu_id, Process)
    for local, gpu_id in enumerate(args.gpus):
        wid = args.worker_offset + local
        procs[wid] = (gpu_id, spawn_worker(
            ctx, gpu_id, wid, args, cu_source,
            job_buf, job_gen, job_lock, out_q, stop_evt))

    # chain poller thread (parent)
    poller = threading.Thread(
        target=chain_poller,
        args=(args, job_buf, job_gen, job_lock, out_q, stop_evt),
        daemon=True,
    )
    poller.start()

    # signal handling
    def _stop(*_):
        log("shutdown requested — stopping workers…")
        stop_evt.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    ready = {}                    # worker_id -> ready info
    banner_sent = False
    seen_nonces = set()
    found_log = open(args.found_log, "a", buffering=1)
    rate_by_gpu = {}              # gpu_id -> MH/s (rolling)
    last_rate_print = time.time()

    def send_banner():
        lines = [f"HTK miner UP — rig {args.rig_id}, {len(args.gpus)} GPU(s)"]
        total = 0.0
        for wid in sorted(ready):
            r = ready[wid]
            total += r["hashrate"]
            lines.append(f"  gpu{r['gpu']} {r['name']}: "
                         f"{r['hashrate']/1e6:.0f} MH/s (block {r['block']}, grid {r['grid']})")
        lines.append(f"  total ~{total/1e6:.0f} MH/s")
        lines.append(f"  workers W={args.worker_offset}..{args.worker_offset+len(args.gpus)-1} of N={args.num_workers}")
        body = "\n".join(lines)
        log(body)
        ntfy_push(args.ntfy_base, args.status_topic, body,
                  title=f"HTK rig {args.rig_id} up", tags="rocket")

    try:
        while not stop_evt.is_set():
            # respawn any dead workers (auto-restart)
            for wid, (gpu_id, p) in list(procs.items()):
                if not p.is_alive() and not stop_evt.is_set():
                    p.join(timeout=1)  # reap the exited process
                    log(f"[gpu{gpu_id}/W{wid}] worker died — restarting")
                    ntfy_push(args.ntfy_base, args.status_topic,
                              f"rig {args.rig_id} gpu{gpu_id} worker restarted",
                              title=f"HTK rig {args.rig_id} warning", tags="warning")
                    procs[wid] = (gpu_id, spawn_worker(
                        ctx, gpu_id, wid, args, cu_source,
                        job_buf, job_gen, job_lock, out_q, stop_evt))

            try:
                msg = out_q.get(timeout=1.0)
            except queue.Empty:
                msg = None

            if msg:
                t = msg["type"]
                if t == "ready":
                    ready[msg["worker"]] = msg
                    if not banner_sent and len(ready) >= len(args.gpus):
                        send_banner()
                        banner_sent = True
                elif t == "found":
                    nonce = msg["nonce"]
                    if nonce in seen_nonces:
                        continue
                    seen_nonces.add(nonce)
                    rec = {"ts": time.time(), "nonce": nonce,
                           "prev_hash": msg["prev_hash"], "gpu": msg["gpu"],
                           "worker": msg["worker"]}
                    found_log.write(json.dumps(rec) + "\n")
                    log(f"*** FOUND nonce 0x{nonce} (gpu{msg['gpu']}) ***")
                    if args.dry_run:
                        log("  --dry-run: not pushing to ntfy")
                    else:
                        ok = ntfy_push(args.ntfy_base, args.ntfy_topic, "0x" + nonce)
                        if ok:
                            log(f"  pushed to ntfy topic '{args.ntfy_topic}'")
                        else:
                            log("  ntfy push FAILED (saved locally; will not retry)")
                            ntfy_push(args.ntfy_base, args.status_topic,
                                      f"rig {args.rig_id}: ntfy nonce push failed (nonce saved locally)",
                                      title=f"HTK rig {args.rig_id} warning", tags="warning")
                elif t == "stat":
                    mhs = msg["hashes"] / max(msg["dt"], 1e-9) / 1e6
                    rate_by_gpu[msg["gpu"]] = mhs
                elif t == "error":
                    log(f"[gpu{msg['gpu']}/W{msg['worker']}] ERROR: {msg['msg']}")

            # aggregate hashrate print every ~10s (local only)
            if time.time() - last_rate_print >= 10.0 and rate_by_gpu:
                total = sum(rate_by_gpu.values())
                per = " ".join(f"g{g}={r:.0f}" for g, r in sorted(rate_by_gpu.items()))
                log(f"[rate] total {total/1000:.2f} GH/s  ({per} MH/s)")
                last_rate_print = time.time()
    finally:
        stop_evt.set()
        for wid, (gpu_id, p) in procs.items():
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        found_log.close()
        log("stopped.")


# ════════════════════════════════════════════════════════════════════════════════
# Self-test: prove GPU output == CPU keccak, exactly like listener.py verifies.
# ════════════════════════════════════════════════════════════════════════════════

# Round constants / rotation offsets / pi-lane permutation for Keccak-f[1600].
_KECCAK_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
_KECCAK_PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]
_KECCAK_ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
_M64 = (1 << 64) - 1


def keccak256(data: bytes) -> bytes:
    """Pure-Python Keccak-256 (Ethereum's, NOT NIST SHA3) — used only by --self-test,
    so the miner's sole runtime deps stay cupy + requests. Standard sponge, rate 136."""
    def rol(x, n):
        return ((x << n) | (x >> (64 - n))) & _M64

    s = [0] * 25
    msg = bytearray(data)
    msg.append(0x01)                      # Keccak pad10*1 (0x01 domain byte)
    while len(msg) % 136 != 0:
        msg.append(0x00)
    msg[-1] |= 0x80
    for off in range(0, len(msg), 136):
        for i in range(136 // 8):
            s[i] ^= int.from_bytes(msg[off + i * 8: off + i * 8 + 8], "little")
        for rnd in range(24):             # Keccak-f[1600]
            c = [s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20] for i in range(5)]
            for i in range(5):
                t = c[(i + 4) % 5] ^ rol(c[(i + 1) % 5], 1)
                for j in range(0, 25, 5):
                    s[i + j] ^= t
            temp = s[1]
            for i in range(24):
                j = _KECCAK_PILN[i]
                t = s[j]
                s[j] = rol(temp, _KECCAK_ROTC[i])
                temp = t
            for j in range(0, 25, 5):
                cc = [s[j + i] for i in range(5)]
                for i in range(5):
                    s[j + i] ^= (~cc[(i + 1) % 5]) & cc[(i + 2) % 5]
            s[0] ^= _KECCAK_RC[rnd]
    out = bytearray()
    for i in range(4):
        out += (s[i] & _M64).to_bytes(8, "little")
    return bytes(out)


def self_test(args, cu_source):
    import cupy as cp
    import numpy as np

    gpu_id = args.gpus[0]
    cp.cuda.Device(gpu_id).use()
    kern = _compile_kernel(cu_source)

    # A real prev_hash if the chain is reachable, else a fixed test value.
    try:
        prev, _ = read_state(args.read_rpc)
        log(f"self-test using live prev_hash 0x{prev.hex()}")
    except Exception as e:
        prev = bytes([0x11]) * 32
        log(f"self-test using fixed prev_hash (chain unreachable: {e})")

    target_int = (1 << 248) - 1                # loose => frequent hits
    target = target_int.to_bytes(32, "big")

    d_prev = cp.asarray(np.frombuffer(prev, dtype=np.uint8).copy())
    d_target = cp.asarray(np.frombuffer(target, dtype=np.uint8).copy())
    d_nonce = cp.zeros(32, dtype=cp.uint8)
    d_flag = cp.zeros(1, dtype=cp.int32)

    block, grid = 256, 4096
    hits = 0
    fails = 0
    bases = []
    for k in range(200):
        base = k
        bases.append(base)
        d_flag.fill(0)
        kern((grid,), (block,), (d_prev, d_target, d_nonce, np.uint64(base), d_flag))
        if int(d_flag.get()[0]):
            nonce = bytes(cp.asnumpy(d_nonce))
            digest = keccak256(nonce + prev)
            val = int.from_bytes(digest, "big")
            ok = val <= target_int
            # cross-check nonce layout: bytes 0-7 == base (LE), bytes 16-31 == 0
            base_ok = int.from_bytes(nonce[0:8], "little") == base
            tail_ok = nonce[16:32] == b"\x00" * 16
            if ok and base_ok and tail_ok:
                hits += 1
            else:
                fails += 1
                log(f"  MISMATCH k={k} nonce=0x{nonce.hex()} val<=t={ok} "
                    f"base_ok={base_ok} tail_ok={tail_ok}")
        if hits >= 25:
            break

    # non-overlap sanity for a 2-worker split (W in {0,1}, N=2)
    s0 = {0 + i * 2 for i in range(1000)}
    s1 = {1 + i * 2 for i in range(1000)}
    disjoint = s0.isdisjoint(s1)

    log(f"self-test: {hits} GPU hits CPU-verified, {fails} mismatches; "
        f"strided-partition disjoint={disjoint}")
    if fails == 0 and hits > 0 and disjoint:
        log("SELF-TEST PASSED ✓")
        return 0
    log("SELF-TEST FAILED ✗")
    return 1


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

def detect_gpu_count():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def parse_gpus(spec):
    if not spec or spec == "all":
        n = detect_gpu_count()
        return list(range(n))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def build_args():
    p = argparse.ArgumentParser(
        description="Multi-GPU HTK CUDA miner for vast.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # fleet / partitioning
    p.add_argument("--rig-id", type=int, default=0,
                   help="this rig's index (0-based) among all rigs")
    p.add_argument("--total-rigs", type=int, default=1,
                   help="number of independent rigs in the fleet")
    p.add_argument("--gpus", default="all",
                   help="'all' or comma list of GPU ids, e.g. 0,1,2,3")
    p.add_argument("--gpus-per-rig", type=int, default=0,
                   help="override GPUs/rig for the partition (0 = use this rig's count)")
    p.add_argument("--global-workers", type=int, default=0,
                   help="override total global workers N (0 = total_rigs*gpus_per_rig)")
    p.add_argument("--worker-offset", type=int, default=-1,
                   help="override this rig's first global worker id (default rig_id*gpus_per_rig)")
    # tuning
    p.add_argument("--target-launch-seconds", type=float, default=2.0,
                   help="grow grid until each kernel launch lasts ~this long")
    p.add_argument("--block-sizes", default="128,256,512,1024",
                   help="candidate block sizes to benchmark")
    p.add_argument("--benchmark-seconds", type=float, default=0.5,
                   help="(reserved) per-config benchmark budget")
    p.add_argument("--retune", action="store_true",
                   help="ignore the tune cache and re-benchmark")
    # chain
    p.add_argument("--read-rpc", default=DEFAULT_READ_RPC,
                   help="Ethereum JSON-RPC for HTK reads")
    p.add_argument("--poll-interval", type=float, default=12.0,
                   help="seconds between prev_hash/max_value polls")
    # output
    p.add_argument("--ntfy-base", default=DEFAULT_NTFY_BASE)
    p.add_argument("--ntfy-topic", default=DEFAULT_NTFY_TOPIC,
                   help="topic listener.py consumes (nonces only)")
    p.add_argument("--status-topic", default=DEFAULT_STATUS_TOPIC,
                   help="separate topic for startup banner + alerts ('' = local only)")
    p.add_argument("--found-log", default=str(SCRIPT_DIR / "found_nonces.jsonl"))
    p.add_argument("--dry-run", action="store_true",
                   help="find + log locally but never push nonces to ntfy")
    p.add_argument("--self-test", action="store_true",
                   help="prove GPU keccak == CPU keccak on a loose target, then exit")

    args = p.parse_args()
    args.gpus = parse_gpus(args.gpus)
    if not args.gpus:
        sys.exit("No CUDA GPUs detected (or --gpus empty).")
    args.block_sizes = [int(x) for x in args.block_sizes.split(",") if x.strip()]

    gpr = args.gpus_per_rig or len(args.gpus)
    args.num_workers = args.global_workers or (args.total_rigs * gpr)
    if args.worker_offset < 0:
        args.worker_offset = args.rig_id * gpr
    if args.num_workers < args.worker_offset + len(args.gpus):
        sys.exit(f"Bad partition: N={args.num_workers} but this rig needs ids "
                 f"{args.worker_offset}..{args.worker_offset+len(args.gpus)-1}. "
                 f"Increase --total-rigs/--global-workers or fix --rig-id.")
    return args


def main():
    args = build_args()
    if not CU_PATH.exists():
        sys.exit(f"kernel source not found: {CU_PATH}")
    cu_source = CU_PATH.read_text()

    log(f"HTK CUDA miner — rig {args.rig_id}/{args.total_rigs}, GPUs {args.gpus}, "
        f"global workers {args.worker_offset}..{args.worker_offset+len(args.gpus)-1} of {args.num_workers}")

    if args.self_test:
        sys.exit(self_test(args, cu_source))
    run_miner(args, cu_source)


if __name__ == "__main__":
    main()
