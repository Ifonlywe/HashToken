"""HashToken CUDA miner -- next-gen.

Standalone: this file has no dependency on any other project directory. Drop it
next to keccak_ng.cu and run it.

Differences from the original htk_cuda_miner.py in this repo:
  * --rig-id: a STABLE identity (from $VAST_CONTAINERLABEL), where the original
    derived its tag from a random start_base and so changed on every restart.
  * Per-rig nonce region, so a share can be proven to come from this rig.
  * Dual-target kernel: emits "shares" clearing a much easier target, letting a
    controller verify the rig is genuinely hashing. Measured cost on an RTX 5090
    is 0.17% throughput and +4 registers.
  * Batched heartbeat+share reporting to an ntfy topic.

Kernel source: keccak_ng.cu (must sit alongside this file).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import secrets
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CU_PATH = SCRIPT_DIR / "keccak_ng.cu"
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
# --- next-gen additions ----------------------------------------------------
# Share proof: rigs emit hashes clearing a far easier target so the controller
# can verify they are actually hashing. See hashtoken-next-gen/docs/selection.md
SHARE_TOPIC = "htk-share-9c41e07b5d2a"
DEFAULT_SHARE_BITS = 36        # ~1 share per 2**36 hashes = ~5.1/min per 5090
SHARE_CAP = 64                 # device ring buffer slots (expected ~0.17/launch)
MAX_SHARES_PER_BATCH = 40      # ntfy message cap; reservoir-sampled if exceeded
DEFAULT_HEARTBEAT = 120.0      # seconds between heartbeat+share messages
REGION_BITS = 24               # per-rig nonce region, binds shares to a rig
REGION_SHIFT = 40
NTFY_TOPIC = "htk-nonce-a303558a1aa9e6043d43531d"
STATUS_TOPIC = "htk-status-35cb5d56831536e9924deb7b"

TARGET_LAUNCH_SECONDS = 2.0
BLOCK_SIZES = [128, 256, 512, 1024]
POLL_INTERVAL = 12.0
MAX_RESTARTS = 5          # per GPU, before declaring it dead instead of spinning
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
    # The kernel is extern "C", so its symbol is unmangled and name_expressions
    # is unnecessary — passing it would make the nvcc backend reject the module.
    try:
        mod = cp.RawModule(code=cu_source, backend="nvcc",
                           options=("-O3", "-std=c++14") + arch)
        return mod.get_function(KERNEL_NAME)
    except Exception as e_nvcc:
        # NVRTC has no filesystem headers at all: drop every #include and spell
        # out the one stdint type the kernel uses.
        stripped = "\n".join(l for l in cu_source.splitlines()
                             if not l.lstrip().startswith("#include"))
        stripped = stripped.replace("uint64_t", "unsigned long long")
        try:
            # NVRTC only, and deliberately WITHOUT -O3: that is an nvcc-only
            # flag and NVRTC rejects it with NVRTC_ERROR_INVALID_OPTION (5),
            # which breaks every rig on an image without nvcc. NVRTC already
            # optimises by default, and Phase -1 measured this path within
            # 0.03% of nvcc on a 5090 using exactly these options.
            mod = cp.RawModule(code=stripped, backend="nvrtc",
                               options=("-std=c++14",))
            return mod.get_function(KERNEL_NAME)
        except Exception as e_nvrtc:
            raise RuntimeError(f"compile failed (nvcc: {e_nvcc}; nvrtc: {e_nvrtc})")


def _benchmark(cp, kern, d_prev, d_target, d_nonce, d_flag, block, grid):
    import numpy as np
    d_flag.fill(0)
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    _sb = cp.zeros(2 * SHARE_CAP, dtype=cp.uint64)
    _sc = cp.zeros(1, dtype=cp.int32)
    # share_target_hi=0 makes the share path essentially never fire, so the
    # benchmark measures the mining path alone.
    kern((grid,), (block,), (d_prev, d_target, d_nonce, np.uint64(0), d_flag,
                             np.uint64(0), _sb, _sc, np.int32(SHARE_CAP)))
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
    errs = []
    for block in BLOCK_SIZES:
        try:
            _benchmark(cp, kern, d_prev, d_zero, d_nonce, d_flag, block, calib)
            hr = _benchmark(cp, kern, d_prev, d_zero, d_nonce, d_flag, block, calib)
        except Exception as e:
            # e.g. "too many resources requested for launch" on the larger blocks.
            # Skip the size rather than killing the whole worker.
            errs.append(f"block={block}: {type(e).__name__}: {e}")
            log(f"  autotune: block={block} UNUSABLE ({type(e).__name__}: {e})")
            continue
        log(f"  autotune: block={block} -> {hr/1e6:.0f} MH/s")
        if best is None or hr > best[0]:
            best = (hr, block)
    if best is None:
        raise RuntimeError("no usable block size — " + "; ".join(errs))
    hashrate, block = best
    grid = max(calib, int(hashrate * TARGET_LAUNCH_SECONDS) // block)
    grid = max(1, min(grid, max_grid))
    return block, grid, hashrate


def resolve_rig_id(cli_value=None):
    """Stable identity for this rig, unlike the upstream rig_tag which is
    derived from a random start_base and therefore changes on every restart.

    VAST_CONTAINERLABEL is the label the controller set at create time, so this
    gives a direct rig-id -> instance-id -> destroy path with no registration
    handshake."""
    import hashlib as _h
    import re as _re
    import socket as _s
    for v in (cli_value, os.environ.get("HTK_RIG_ID"),
              os.environ.get("VAST_CONTAINERLABEL")):
        if v and str(v).strip() and str(v).strip() != "unknown":
            return _re.sub(r"[^A-Za-z0-9_.-]", "", str(v).strip())[:32]
    return "h" + _h.sha256(_s.gethostname().encode()).hexdigest()[:11]


def region_for(rig_id):
    """24-bit nonce region owned by this rig. Verification rejects any share
    whose base carries a different region, so one rig cannot submit another's
    work as its own."""
    return int.from_bytes(keccak256(rig_id.encode())[:3], "big")


def share_target_hi(share_bits):
    """64-bit share target. The kernel compares only the top 64 bits of the
    digest, so P(hit) = (target+1)/2**64."""
    return (1 << (64 - share_bits)) - 1


def gpu_worker(gpu_id, worker_id, num_workers, cu_source, job_buf, job_gen, job_lock, out_q, stop_evt,
               share_tgt=0, region=0):
    tag = f"gpu{gpu_id}"
    # Names the phase we are in, so a crash says WHERE and not just WHAT.
    step = "startup"
    try:
        step = "import cupy"
        import cupy as cp
        import numpy as np

        step = f"select device {gpu_id}"
        dev = cp.cuda.Device(gpu_id)
        dev.use()

        step = "read device properties"
        raw_name = cp.cuda.runtime.getDeviceProperties(gpu_id)["name"]
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        cc = dev.compute_capability
        log(f"[{tag}] device: {name} sm_{cc}")

        step = "compile kernel"
        kern = _compile_kernel(cu_source)
        step = "allocate device buffers"
        d_prev = cp.zeros(32, dtype=cp.uint8)
        d_target = cp.full(32, 255, dtype=cp.uint8)
        d_nonce = cp.zeros(32, dtype=cp.uint8)
        # One combined control array so the per-launch device->host sync count
        # does NOT increase: [0] = found flag, [1] = share count. The upstream
        # loop already pays exactly one blocking .get() per ~2s launch.
        d_ctl = cp.zeros(2, dtype=cp.int32)
        d_flag = d_ctl[0:1]
        d_scount = d_ctl[1:2]
        d_shares = cp.zeros(2 * SHARE_CAP, dtype=cp.uint64)

        step = "read tune cache"
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
            step = "autotune"
            log(f"[{tag}] autotuning (block sizes {BLOCK_SIZES})…")
            block, grid, hashrate = _autotune(cp, kern, dev, d_prev, d_nonce, d_flag)
            cache[ck] = {"block": block, "grid": grid, "hashrate": hashrate}
            try:
                TUNE_CACHE.write_text(json.dumps(cache, indent=2))
            except Exception:
                pass

        out_q.put({"type": "ready", "gpu": gpu_id, "name": f"{name} (sm_{cc})",
                   "block": block, "grid": grid, "hashrate": hashrate})
        log(f"[{tag}] {name} sm_{cc}: block={block} grid={grid} ~{hashrate/1e6:.1f} MH/s")

        step = "mining loop"
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

            # Region-bound base. The rig owns 2**40 bases x 2**64 thread ids
            # = 2**104 hashes, so exhaustion is not a concern; the point is that
            # the region proves which rig produced a share.
            counter = (worker_id + k * num_workers) & ((1 << REGION_SHIFT) - 1)
            base = ((region << REGION_SHIFT) | counter) & MASK64
            k += 1
            d_ctl.fill(0)
            kern((grid,), (block,), (d_prev, d_target, d_nonce, np.uint64(base), d_flag,
                                     np.uint64(share_tgt), d_shares, d_scount,
                                     np.int32(SHARE_CAP)))
            hashes += block * grid
            ctl = d_ctl.get()                      # single sync, as upstream
            n_sh = int(ctl[1])
            if n_sh:
                keep = min(n_sh, SHARE_CAP)
                raw = cp.asnumpy(d_shares[:2 * keep])
                out_q.put({"type": "shares", "gpu": gpu_id, "prev": prev.hex(),
                           "count": n_sh, "kept": keep,
                           "pairs": [(int(raw[2 * i]), int(raw[2 * i + 1]))
                                     for i in range(keep)]})
            if int(ctl[0]):
                nonce = bytes(cp.asnumpy(d_nonce))
                out_q.put({"type": "found", "gpu": gpu_id, "nonce": nonce.hex(), "prev_hash": prev.hex()})
                d_nonce.fill(0)

            if time.time() - t0 >= 5.0:
                out_q.put({"type": "stat", "gpu": gpu_id, "hashes": hashes, "dt": time.time() - t0})
                hashes = 0
                t0 = time.time()
    except Exception as e:
        # Print first: stdout is inherited from the coordinator, so this reaches
        # the vast console / onstart.log even if the queue hand-off fails.
        print(f"\n===== [{tag}] WORKER CRASHED during: {step} =====", flush=True)
        traceback.print_exc()
        print(f"===== [{tag}] end traceback =====\n", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            out_q.put({"type": "error", "gpu": gpu_id, "msg": f"[{step}] {type(e).__name__}: {e}"})
            # put() only hands off to a feeder thread; os._exit would kill it
            # mid-flush and lose the message — which is why crashes used to be
            # invisible. Wait for the pipe write to actually complete.
            out_q.close()
            out_q.join_thread()
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


def spawn_worker(ctx, gpu_id, worker_id, num_workers, cu_source, job_buf, job_gen, job_lock, out_q, stop_evt,
                 share_tgt=0, region=0):
    p = ctx.Process(target=gpu_worker,
                    args=(gpu_id, worker_id, num_workers, cu_source, job_buf, job_gen, job_lock, out_q, stop_evt,
                          share_tgt, region))
    p.start()
    return p


def run_miner(args, cu_source):
    import cupy as cp
    gpus = list(range(cp.cuda.runtime.getDeviceCount()))
    if not gpus:
        raise SystemExit("No CUDA GPUs detected.")
    num_workers = len(gpus)
    # start_base stays RANDOM. Deriving it from the rig id would make a
    # crash-looping rig re-walk the identical nonce region forever.
    start_base = secrets.randbits(64)
    rig_tag = resolve_rig_id(getattr(args, "rig_id", None))
    share_bits = int(getattr(args, "share_bits", DEFAULT_SHARE_BITS))
    region = (getattr(args, "nonce_region", None)
              if getattr(args, "nonce_region", None) is not None
              else region_for(rig_tag))
    share_tgt = share_target_hi(share_bits)
    # More GPUs means proportionally more shares; raise the difficulty so the
    # per-rig message size stays constant regardless of rig width.
    if num_workers > 1:
        share_bits += max(0, (num_workers - 1).bit_length())
        share_tgt = share_target_hi(share_bits)
    try:
        dph = float(os.environ.get("HTK_DPH") or 0)
    except ValueError:
        dph = 0.0
    log(f"rig={rig_tag} region={region:#08x} share_bits={share_bits} "
        f"dph=${dph:.4f}/hr")
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
                                     job_buf, job_gen, job_lock, out_q, stop_evt,
                                     share_tgt, region)

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
    fails = {g: 0 for g in gpus}     # consecutive crashes per GPU
    last_err = {}                    # last error text reported by each GPU
    found_log = open(FOUND_LOG, "a", buffering=1)
    last_print = time.time()
    reporter = ShareReporter(rig_tag, region, share_bits,
                             getattr(args, "share_topic", SHARE_TOPIC),
                             float(getattr(args, "heartbeat_secs", DEFAULT_HEARTBEAT)),
                             gpus={})

    def send_banner():
        """Compact status: what this box is, what it costs, and -- the number
        that actually decides whether to keep it -- what one HTK would cost if
        this machine mined it alone at the current on-chain difficulty."""
        total = sum(r["hashrate"] for r in ready.values())
        name = ready[min(ready)]["name"] if ready else "?"
        name = name.split(" (")[0].replace("NVIDIA GeForce ", "")

        # max_value is the live chain target, written by chain_poller into the
        # second half of job_buf. Difficulty = expected hashes per coin.
        with job_lock:
            mv = int.from_bytes(bytes(job_buf[32:64]), "big")
        cost = ""
        if mv and total > 0:
            hashes = (1 << 256) // (mv + 1)
            hours = hashes / total / 3600.0
            line = f"{hours:,.0f}h/coin @ {hashes/1e12:,.0f} TH"
            if dph > 0:
                line += f" -> ${hours * dph:,.2f}/HTK"
            cost = line
        elif total > 0:
            cost = "awaiting chain difficulty"

        body = (f"{name} x{len(gpus)} | {total/1e9:.2f} GH/s"
                + (f" | ${dph:.4f}/hr" if dph > 0 else "")
                + (f"\n{cost}" if cost else ""))
        log(f"UP {rig_tag}: {body}")
        ntfy_push(args.status_topic, body, title=f"HTK {rig_tag} up", tags="rocket")

    try:
        while not stop_evt.is_set():
            # Drain every queued message BEFORE reacting to a death, so a
            # worker's dying error is on record when we report the crash.
            msgs = []
            try:
                msgs.append(out_q.get(timeout=1.0))
            except queue.Empty:
                pass
            while True:
                try:
                    msgs.append(out_q.get_nowait())
                except queue.Empty:
                    break

            for msg in msgs:
                t = msg["type"]
                if t == "ready":
                    ready[msg["gpu"]] = msg
                    fails[msg["gpu"]] = 0          # reached steady state, reset
                    reporter.gpus[msg["gpu"]] = msg["name"]
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
                elif t == "shares":
                    reporter.add(msg)
                elif t == "stat":
                    rate[msg["gpu"]] = msg["hashes"] / max(msg["dt"], 1e-9) / 1e6
                    reporter.rates[msg["gpu"]] = rate[msg["gpu"]]
                elif t == "error":
                    last_err[msg["gpu"]] = msg["msg"]
                    log(f"[gpu{msg['gpu']}] ERROR: {msg['msg']}")

            for gpu_id, p in list(procs.items()):
                if p.is_alive() or stop_evt.is_set():
                    continue
                p.join(timeout=1)
                fails[gpu_id] += 1
                n = fails[gpu_id]
                err = last_err.get(gpu_id, "no error captured — see the worker traceback above")

                if n > MAX_RESTARTS:
                    log(f"[gpu{gpu_id}] GIVING UP after {MAX_RESTARTS} restarts — {err}")
                    ntfy_push(args.status_topic,
                              f"rig {rig_tag} gpu{gpu_id} DEAD after {MAX_RESTARTS} restarts\n{err}",
                              title=f"HTK rig {rig_tag} gpu{gpu_id} dead", tags="rotating_light")
                    procs.pop(gpu_id, None)
                    if not procs:
                        log("all GPU workers dead — exiting so the rig stops burning time")
                        stop_evt.set()
                    continue

                delay = min(2 ** n, 60)
                log(f"[gpu{gpu_id}] worker died ({n}/{MAX_RESTARTS}) — retry in {delay}s — {err}")
                if n == 1:      # alert once, not on every retry
                    ntfy_push(args.status_topic, f"rig {rig_tag} gpu{gpu_id} crashed: {err}",
                              title=f"HTK rig {rig_tag} warning", tags="warning")
                stop_evt.wait(delay)
                if stop_evt.is_set():
                    break
                wid = secrets.randbits(64)
                reporter.restarts += 1
                procs[gpu_id] = spawn_worker(ctx, gpu_id, wid, num_workers, cu_source,
                                             job_buf, job_gen, job_lock, out_q, stop_evt,
                                             share_tgt, region)

            if time.time() - last_print >= 10.0 and rate:
                total = sum(rate.values())
                per = " ".join(f"g{g}={r:.0f}" for g, r in sorted(rate.items()))
                log(f"[rate] total {total/1000:.2f} GH/s  ({per} MH/s)")
                last_print = time.time()

            # Heartbeat and shares travel together, so "heartbeat with no
            # shares" is directly observable by the controller.
            if reporter.due():
                ok, m = reporter.flush()
                log(f"[hb] {len(m['sh'])} shares, sc={m['sc']}, "
                    f"published={ok}")
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
    d_shares = cp.zeros(2 * SHARE_CAP, dtype=cp.uint64)
    d_scount = cp.zeros(1, dtype=cp.int32)

    # Share leg: run at an easy target so shares actually appear, then
    # CPU-verify every pair. This validates the exact bytes the controller will
    # trust -- above all the nonce reconstruction, where the kernel writes two
    # little-endian u64s but the digest comparison is big-endian. That mismatch
    # is the likeliest bug in the whole design and it is invisible until a
    # verifier rejects everything.
    st_bits = 20
    st_tgt = share_target_hi(st_bits)
    sh_ok = sh_bad = 0
    for k in range(20):
        d_scount.fill(0)
        d_flag.fill(0)
        kern((4096,), (256,), (d_prev, d_target, d_nonce, np.uint64(k), d_flag,
                               np.uint64(st_tgt), d_shares, d_scount,
                               np.int32(SHARE_CAP)))
        n = int(d_scount.get()[0])
        if not n:
            continue
        keep = min(n, SHARE_CAP)
        raw = cp.asnumpy(d_shares[:2 * keep])
        for i in range(keep):
            b, t = int(raw[2 * i]), int(raw[2 * i + 1])
            nonce = (b & MASK64).to_bytes(8, "little") + \
                    (t & MASK64).to_bytes(8, "little") + b"\x00" * 16
            if int.from_bytes(keccak256(nonce + prev), "big") >> 192 <= st_tgt:
                sh_ok += 1
            else:
                sh_bad += 1
                log(f"  SHARE MISMATCH base={b:#x} tid={t:#x}")
        if sh_ok >= 40:
            break
    log(f"self-test shares: {sh_ok} CPU-verified, {sh_bad} mismatches "
        f"(target 2**-{st_bits})")

    hits = fails = 0
    for k in range(200):
        d_flag.fill(0)
        d_scount.fill(0)
        kern((4096,), (256,), (d_prev, d_target, d_nonce, np.uint64(k), d_flag,
                               np.uint64(0), d_shares, d_scount, np.int32(SHARE_CAP)))
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

    # Exercise the worker's SETUP path too. The hash check above runs a fixed
    # block=256/grid=4096 launch in this process and never touches device
    # properties or the autotune sweep — which is exactly where workers die.
    tune_ok = True
    try:
        dev = cp.cuda.Device(0)
        raw_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        dev_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        log(f"self-test device: {dev_name} sm_{dev.compute_capability}")
        b, g, hr = _autotune(cp, kern, dev, d_prev, d_nonce, d_flag)
        log(f"self-test autotune: block={b} grid={g} ~{hr/1e6:.0f} MH/s")
    except Exception as e:
        tune_ok = False
        log(f"self-test autotune FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

    disjoint = {i * 2 for i in range(1000)}.isdisjoint({1 + i * 2 for i in range(1000)})
    log(f"self-test: {hits} GPU hits CPU-verified, {fails} mismatches; "
        f"autotune={'ok' if tune_ok else 'FAILED'}; disjoint={disjoint}")
    if sh_bad or sh_ok == 0:
        log("SELF-TEST FAILED - share path broken")
        return 1
    if fails == 0 and hits > 0 and disjoint and tune_ok:
        log("SELF-TEST PASSED ✓")
        return 0
    log("SELF-TEST FAILED ✗")
    return 1


# ===========================================================================
# NEXT-GEN: share batching and heartbeat reporting
# ===========================================================================

class ShareReporter:
    """Collect shares from the workers and publish one message per interval.

    Heartbeat and shares are the SAME message on purpose: it halves ntfy volume
    and makes "heartbeat arrived but carried no shares" a directly observable
    state rather than something the controller has to infer by correlating two
    streams.
    """

    def __init__(self, rig_id, region, share_bits, topic, interval, gpus):
        self.rig_id = rig_id
        self.region = region
        self.share_bits = share_bits
        self.topic = topic
        self.interval = interval
        self.gpus = gpus
        self.t0 = time.time()
        self.last = time.time()
        self.pending = []          # (prev_hex, base, tid)
        self.seen_total = 0        # true hit count incl. buffer overflow
        self.restarts = 0
        self.found = 0
        self.rates = {}

    def add(self, msg):
        self.seen_total += int(msg.get("count", 0))
        prev = msg["prev"]
        for b, t in msg["pairs"]:
            self.pending.append((prev, b, t))

    def due(self, now=None):
        return (now or time.time()) - self.last >= self.interval

    def build(self, now=None):
        now = now or time.time()
        sample = self.pending
        if len(sample) > MAX_SHARES_PER_BATCH:
            # Reservoir sample so the transmitted subset stays unbiased; a
            # biased sample would skew the controller's hashrate estimate.
            sample = random.sample(sample, MAX_SHARES_PER_BATCH)
        prevs, idx = [], {}
        sh = []
        for prev, b, t in sample:
            if prev not in idx:
                idx[prev] = len(prevs)
                prevs.append(prev)
            sh.append([idx[prev], f"{b:016x}", f"{t:016x}"])
        return {
            "v": 1, "rig": self.rig_id, "t": int(now),
            "up": int(now - self.t0), "rst": self.restarts,
            "region": self.region, "sb": self.share_bits,
            "gpus": [{"i": g, "n": n, "hr": round(self.rates.get(g, 0.0), 1)}
                     for g, n in sorted(self.gpus.items())],
            "prevs": prevs, "sh": sh,
            "sc": self.seen_total, "nf": self.found, "err": None,
        }

    def flush(self, now=None):
        msg = self.build(now)
        ok = ntfy_push(self.topic, json.dumps(msg))
        self.pending = []
        self.seen_total = 0
        self.last = now or time.time()
        return ok, msg


def main():
    p = argparse.ArgumentParser(description="Multi-GPU HTK CUDA miner")
    p.add_argument("--status-topic", default=STATUS_TOPIC)
    p.add_argument("--rig-id", default=None,
                   help="stable identity; defaults to $VAST_CONTAINERLABEL")
    p.add_argument("--share-topic", default=SHARE_TOPIC)
    p.add_argument("--share-bits", type=int, default=DEFAULT_SHARE_BITS)
    p.add_argument("--heartbeat-secs", type=float, default=DEFAULT_HEARTBEAT)
    p.add_argument("--nonce-region", type=int, default=None,
                   help="override the derived region (testing only)")
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
