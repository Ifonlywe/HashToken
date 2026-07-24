#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import secrets
import signal
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CU_PATH = SCRIPT_DIR / "keccak_miner.cu"
KERNEL_NAME = "find_solution_kernel"
MASK64 = (1 << 64) - 1

CONTRACT = "0xE5544a2A5fA9b175da60D8Eec67adD5582bB31b0"
SEL_PREV_HASH = "0xc69b5df2"
SEL_MAX_VALUE = "0x98597629"
READ_RPCS = [
    "https://ethereum-rpc.publicnode.com",
    "https://eth.llamarpc.com",
    "https://cloudflare-eth.com",
    "https://eth.drpc.org",
    "https://1rpc.io/eth",
]

NTFY_BASE = "https://ntfy.sh"
NTFY_TOPIC = "htk-nonce-a303558a1aa9e6043d43531d"
STATUS_TOPIC = "htk-status-35cb5d56831536e9924deb7b"

TARGET_LAUNCH_SECONDS = 2.0
BLOCK_SIZES = [128, 256, 512, 1024]
POLL_INTERVAL = 12.0
FOUND_LOG = SCRIPT_DIR / "found_nonces.jsonl"
TUNE_CACHE = SCRIPT_DIR / ".tune_cache.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _compile_kernel(cu_source):
    import cupy as cp
    try:
        cc = cp.cuda.Device().compute_capability
        arch = (f"-arch=sm_{cc}",)
    except Exception:
        arch = ()
    try:
        mod = cp.RawModule(code=cu_source, backend="nvcc",
                           options=("-O3", "-std=c++14") + arch,
                           name_expressions=[KERNEL_NAME])
        return mod.get_function(KERNEL_NAME)
    except Exception as e_nvcc:
        stripped = "\n".join(l for l in cu_source.splitlines() if "cuda_runtime.h" not in l)
        try:
            mod = cp.RawModule(code=stripped, backend="nvrtc",
                               options=("-std=c++14",), name_expressions=[KERNEL_NAME])
            return mod.get_function(KERNEL_NAME)
        except Exception as e_nvrtc:
            raise RuntimeError(f"compile failed (nvcc: {e_nvcc}; nvrtc: {e_nvrtc})")


def _benchmark(cp, kern, d_prev, d_target, d_nonce, d_flag, block, grid):
    import numpy as np
    d_flag.fill(0)
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    kern((grid,), (block,), (d_prev, d_target, d_nonce, np.uint64(0), d_flag))
    end.record()
    end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end)
    return (block * grid) / (ms / 1000.0) if ms > 0 else 0.0


def _autotune(cp, kern, dev, d_prev, d_nonce, d_flag):
    d_zero = cp.zeros(32, dtype=cp.uint8)
    sm = dev.attributes.get("MultiProcessorCount", 16)
    max_grid = int(dev.attributes.get("MaxGridDimX", 2**31 - 1))
    calib = sm * 64
    best = None
    for block in BLOCK_SIZES:
        _benchmark(cp, kern, d_prev, d_zero, d_nonce, d_flag, block, calib)
        hr = _benchmark(cp, kern, d_prev, d_zero, d_nonce, d_flag, block, calib)
        if best is None or hr > best[0]:
            best = (hr, block)
    hashrate, block = best
    grid = max(calib, int(hashrate * TARGET_LAUNCH_SECONDS) // block)
    grid = max(1, min(grid, max_grid))
    return block, grid, hashrate


def gpu_worker(gpu_id, worker_id, num_workers, cu_source, job_buf, job_gen, job_lock, out_q, stop_evt):
    tag = f"gpu{gpu_id}"
    try:
        import cupy as cp
        import numpy as np

        dev = cp.cuda.Device(gpu_id)
        dev.use()
        name = cp.cuda.runtime.getDeviceProperties(gpu_id)["name"].decode()
        cc = dev.compute_capability

        kern = _compile_kernel(cu_source)
        d_prev = cp.zeros(32, dtype=cp.uint8)
        d_target = cp.full(32, 255, dtype=cp.uint8)
        d_nonce = cp.zeros(32, dtype=cp.uint8)
        d_flag = cp.zeros(1, dtype=cp.int32)

        cache = {}
        if TUNE_CACHE.exists():
            try:
                cache = json.loads(TUNE_CACHE.read_text())
            except Exception:
                cache = {}
        ck = f"{name}|{TARGET_LAUNCH_SECONDS}"
        if ck in cache:
            block, grid, hashrate = cache[ck]["block"], cache[ck]["grid"], cache[ck].get("hashrate", 0.0)
        else:
            block, grid, hashrate = _autotune(cp, kern, dev, d_prev, d_nonce, d_flag)
            cache[ck] = {"block": block, "grid": grid, "hashrate": hashrate}
            try:
                TUNE_CACHE.write_text(json.dumps(cache, indent=2))
            except Exception:
                pass

        out_q.put({"type": "ready", "gpu": gpu_id, "name": f"{name} (sm_{cc})",
                   "block": block, "grid": grid, "hashrate": hashrate})
        log(f"[{tag}] {name} sm_{cc}: block={block} grid={grid} ~{hashrate/1e6:.1f} MH/s")

        k = 0
        cur_gen = -1
        prev = b"\x00" * 32
        have_job = False
        hashes = 0
        t0 = time.time()

        while not stop_evt.is_set():
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
            hashes += block * grid
            if int(d_flag.get()[0]):
                nonce = bytes(cp.asnumpy(d_nonce))
                out_q.put({"type": "found", "gpu": gpu_id, "nonce": nonce.hex(), "prev_hash": prev.hex()})
                d_nonce.fill(0)

            if time.time() - t0 >= 5.0:
                out_q.put({"type": "stat", "gpu": gpu_id, "hashes": hashes, "dt": time.time() - t0})
                hashes = 0
                t0 = time.time()
    except Exception as e:
        try:
            out_q.put({"type": "error", "gpu": gpu_id, "msg": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
        os._exit(1)


def ntfy_push(topic, body, title=None, tags=None, tries=4):
    import requests
    if not topic:
        return False
    headers = {}
    if title:
        headers["Title"] = title
    if tags:
        headers["Tags"] = tags
    delay = 1.0
    for _ in range(tries):
        try:
            requests.post(f"{NTFY_BASE}/{topic}", data=body.encode(), headers=headers, timeout=15).raise_for_status()
            return True
        except Exception:
            time.sleep(delay)
            delay = min(delay * 2, 15)
    return False


def _eth_call(rpc, data, timeout=10):
    import requests
    r = requests.post(rpc, json={"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                                 "params": [{"to": CONTRACT, "data": data}, "latest"]}, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(body["error"])
    result = body.get("result", "0x")
    return bytes.fromhex(result[2:] if result.startswith("0x") else result)


def read_state(rpc):
    prev = _eth_call(rpc, SEL_PREV_HASH)
    mx = _eth_call(rpc, SEL_MAX_VALUE)
    if len(prev) != 32 or len(mx) != 32:
        raise ValueError("bad return length")
    return prev, int.from_bytes(mx, "big")


def read_state_failover(start=0):
    errs = []
    n = len(READ_RPCS)
    for i in range(n):
        idx = (start + i) % n
        try:
            prev, mx = read_state(READ_RPCS[idx])
            return prev, mx, idx
        except Exception as e:
            errs.append(f"{READ_RPCS[idx]}: {type(e).__name__}")
    raise RuntimeError("all read RPCs failed — " + "; ".join(errs))


def chain_poller(job_buf, job_gen, job_lock, out_q, stop_evt):
    cur = 0
    last_prev = None
    last_rpc = None
    backoff = 1.0
    while not stop_evt.is_set():
        try:
            prev, mx, cur = read_state_failover(cur)
            backoff = 1.0
            if READ_RPCS[cur] != last_rpc:
                if last_rpc is not None:
                    log(f"[rpc] switched to {READ_RPCS[cur]}")
                last_rpc = READ_RPCS[cur]
            if prev != last_prev:
                with job_lock:
                    job_buf[0:32] = prev
                    job_buf[32:64] = mx.to_bytes(32, "big")
                    job_gen.value += 1
                last_prev = prev
                log(f"[job] prev_hash=0x{prev.hex()[:16]}… (search restarted)")
        except Exception as e:
            out_q.put({"type": "error", "gpu": -1, "msg": f"all read RPCs down ({e}); mining continues"})
            time.sleep(backoff)
            backoff = min(backoff * 2, POLL_INTERVAL)
        stop_evt.wait(POLL_INTERVAL)


def spawn_worker(ctx, gpu_id, worker_id, num_workers, cu_source, job_buf, job_gen, job_lock, out_q, stop_evt):
    p = ctx.Process(target=gpu_worker,
                    args=(gpu_id, worker_id, num_workers, cu_source, job_buf, job_gen, job_lock, out_q, stop_evt))
    p.start()
    return p


def run_miner(args, cu_source):
    import cupy as cp
    gpus = list(range(cp.cuda.runtime.getDeviceCount()))
    if not gpus:
        raise SystemExit("No CUDA GPUs detected.")
    num_workers = len(gpus)
    start_base = secrets.randbits(64)
    rig_tag = f"{start_base:016x}"[:8]
    log(f"HTK CUDA miner — GPUs {gpus}, random start base=0x{start_base:016x} stride={num_workers}")

    ctx = mp.get_context("spawn")
    job_buf = ctx.Array("c", 64, lock=False)
    job_lock = ctx.Lock()
    job_gen = ctx.Value("l", 0, lock=False)
    out_q = ctx.Queue()
    stop_evt = ctx.Event()

    procs = {}
    for local, gpu_id in enumerate(gpus):
        wid = (start_base + local) & MASK64
        procs[gpu_id] = spawn_worker(ctx, gpu_id, wid, num_workers, cu_source,
                                     job_buf, job_gen, job_lock, out_q, stop_evt)

    threading.Thread(target=chain_poller,
                     args=(job_buf, job_gen, job_lock, out_q, stop_evt), daemon=True).start()

    def _stop(*_):
        log("shutdown requested — stopping workers…")
        stop_evt.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    ready = {}
    banner_sent = False
    seen = set()
    rate = {}
    found_log = open(FOUND_LOG, "a", buffering=1)
    last_print = time.time()

    def send_banner():
        total = sum(r["hashrate"] for r in ready.values())
        lines = [f"HTK miner UP — rig {rig_tag}, {len(gpus)} GPU(s)"]
        for g in sorted(ready):
            r = ready[g]
            lines.append(f"  gpu{g} {r['name']}: {r['hashrate']/1e6:.0f} MH/s")
        lines.append(f"  total ~{total/1e6:.0f} MH/s")
        body = "\n".join(lines)
        log(body)
        ntfy_push(args.status_topic, body, title=f"HTK rig {rig_tag} up", tags="rocket")

    try:
        while not stop_evt.is_set():
            for gpu_id, p in list(procs.items()):
                if not p.is_alive() and not stop_evt.is_set():
                    p.join(timeout=1)
                    log(f"[gpu{gpu_id}] worker died — restarting")
                    ntfy_push(args.status_topic, f"rig {rig_tag} gpu{gpu_id} restarted",
                              title=f"HTK rig {rig_tag} warning", tags="warning")
                    wid = secrets.randbits(64)
                    procs[gpu_id] = spawn_worker(ctx, gpu_id, wid, num_workers, cu_source,
                                                 job_buf, job_gen, job_lock, out_q, stop_evt)
            try:
                msg = out_q.get(timeout=1.0)
            except queue.Empty:
                msg = None

            if msg:
                t = msg["type"]
                if t == "ready":
                    ready[msg["gpu"]] = msg
                    if not banner_sent and len(ready) >= len(gpus):
                        send_banner()
                        banner_sent = True
                elif t == "found":
                    nonce = msg["nonce"]
                    if nonce in seen:
                        continue
                    seen.add(nonce)
                    found_log.write(json.dumps({"ts": time.time(), "nonce": nonce,
                                                "prev_hash": msg["prev_hash"], "gpu": msg["gpu"]}) + "\n")
                    log(f"*** FOUND nonce 0x{nonce} (gpu{msg['gpu']}) ***")
                    if args.dry_run:
                        log("  --dry-run: not pushing")
                    elif ntfy_push(NTFY_TOPIC, "0x" + nonce):
                        log("  pushed to ntfy")
                    else:
                        log("  ntfy push FAILED (saved locally)")
                        ntfy_push(args.status_topic, f"rig {rig_tag}: ntfy nonce push failed (saved locally)",
                                  title=f"HTK rig {rig_tag} warning", tags="warning")
                elif t == "stat":
                    rate[msg["gpu"]] = msg["hashes"] / max(msg["dt"], 1e-9) / 1e6
                elif t == "error":
                    log(f"[gpu{msg['gpu']}] ERROR: {msg['msg']}")

            if time.time() - last_print >= 10.0 and rate:
                total = sum(rate.values())
                per = " ".join(f"g{g}={r:.0f}" for g, r in sorted(rate.items()))
                log(f"[rate] total {total/1000:.2f} GH/s  ({per} MH/s)")
                last_print = time.time()
    finally:
        stop_evt.set()
        for p in procs.values():
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        found_log.close()
        log("stopped.")


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


def keccak256(data):
    def rol(x, n):
        return ((x << n) | (x >> (64 - n))) & MASK64
    s = [0] * 25
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % 136 != 0:
        msg.append(0x00)
    msg[-1] |= 0x80
    for off in range(0, len(msg), 136):
        for i in range(17):
            s[i] ^= int.from_bytes(msg[off + i * 8: off + i * 8 + 8], "little")
        for rnd in range(24):
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
        out += (s[i] & MASK64).to_bytes(8, "little")
    return bytes(out)


def self_test(cu_source):
    import cupy as cp
    import numpy as np

    cp.cuda.Device(0).use()
    kern = _compile_kernel(cu_source)
    try:
        prev, _mx, _idx = read_state_failover()
        log(f"self-test using live prev_hash 0x{prev.hex()}")
    except Exception as e:
        prev = bytes([0x11]) * 32
        log(f"self-test using fixed prev_hash (chain unreachable: {e})")

    target_int = (1 << 248) - 1
    d_prev = cp.asarray(np.frombuffer(prev, dtype=np.uint8).copy())
    d_target = cp.asarray(np.frombuffer(target_int.to_bytes(32, "big"), dtype=np.uint8).copy())
    d_nonce = cp.zeros(32, dtype=cp.uint8)
    d_flag = cp.zeros(1, dtype=cp.int32)

    hits = fails = 0
    for k in range(200):
        d_flag.fill(0)
        kern((4096,), (256,), (d_prev, d_target, d_nonce, np.uint64(k), d_flag))
        if int(d_flag.get()[0]):
            nonce = bytes(cp.asnumpy(d_nonce))
            val = int.from_bytes(keccak256(nonce + prev), "big")
            if val <= target_int and int.from_bytes(nonce[0:8], "little") == k and nonce[16:32] == b"\x00" * 16:
                hits += 1
            else:
                fails += 1
                log(f"  MISMATCH k={k} nonce=0x{nonce.hex()}")
        if hits >= 25:
            break

    disjoint = {i * 2 for i in range(1000)}.isdisjoint({1 + i * 2 for i in range(1000)})
    log(f"self-test: {hits} GPU hits CPU-verified, {fails} mismatches; disjoint={disjoint}")
    if fails == 0 and hits > 0 and disjoint:
        log("SELF-TEST PASSED ✓")
        return 0
    log("SELF-TEST FAILED ✗")
    return 1


def main():
    p = argparse.ArgumentParser(description="Multi-GPU HTK CUDA miner")
    p.add_argument("--status-topic", default=STATUS_TOPIC)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if not CU_PATH.exists():
        raise SystemExit(f"kernel source not found: {CU_PATH}")
    cu_source = CU_PATH.read_text()

    if args.self_test:
        raise SystemExit(self_test(cu_source))
    run_miner(args, cu_source)


if __name__ == "__main__":
    main()
