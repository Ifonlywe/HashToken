#include <cuda_runtime.h>
#include <stdint.h>

// ===========================================================================
// HashToken mining kernel -- optimized drop-in replacement for keccak_ng.cu.
//
// DROP-IN CONTRACT. htk_miner_ng.py resolves the kernel by name and passes
// nine POSITIONAL arguments (see htk_miner_ng.py:272). This file keeps that
// name and that argument list byte-for-byte, and keeps one hash per thread
// with nonce = (base_nonce_part, threadIdx-derived id), so it can replace
// keccak_ng.cu with no host-side change whatsoever.
//
// Semantics are identical to the kernel it replaces: same nonce layout, same
// Keccak-256 (0x01 pad -- Ethereum, NOT the FIPS-202 0x06), same big-endian
// hash <= target test with equality counting as a solution, same share
// ring-buffer protocol including atomicAdd-returns-pre-increment so the host
// can still distinguish "buffer overflowed" from "this rig is lying".
//
// What is faster, and why:
//   1. No byte arrays. The old kernel built a 64-byte message one byte at a
//      time, cast it to uint64*, hashed, wrote four lanes into a 32-byte array
//      and read eight of those bytes back one at a time. Lanes are now
//      assigned directly. (Also fixes alignment UB: (const unsigned long long*) on a
//      char[64] is not guaranteed 8-byte aligned.)
//   2. Fully unrolled permutation, literal round constants and literal rho/pi
//      indices. The old rho/pi indexed s[j] with j from a __device__ table; it
//      only stayed in registers while nvcc constant-folded that table.
//   3. 64-bit rotates as two 32-bit funnel shifts.
//   4. Pruned final round. Both target tests read ONLY output lane 0, and
//      lane 0 after round 23 needs just three lanes of the post-theta state.
//      Lanes 1..3 were computed and stored by the old kernel and never read.
//   5. The exact 32-byte compare runs only when the top 64 bits already tie or
//      beat the target, and is __noinline__ so its registers do not weigh on
//      the hot path.
//
// CORRECTNESS UNDER ANY DIFFICULTY. The hot path computes only the top 64 bits
// of the digest, but computes them exactly. Whenever those bits tie or beat
// the target the FULL 256-bit hash is recomputed and compared byte-wise, so a
// solution can never be missed or falsely claimed. If HTK ever got easy enough
// that the top 64 bits stopped deciding, this kernel would get slower, never
// wrong.
//
// Build switches (all default to the fast path):
//   -DHTK_USE_FUNNEL=0    plain shift/or rotate idiom
//   -DHTK_PRUNE_LAST=0    run a full 24th round
//   -DHTK_KERNEL_NAME=x   rename the entry point (used by the benchmark only)
// ===========================================================================

#ifndef HTK_KERNEL_NAME
#define HTK_KERNEL_NAME find_solution_kernel
#endif
// Funnel-shift rotates are a Blackwell-and-later win and a small LOSS before
// that. Measured on 2^26 nonces, best of 5, block 256:
//   sm_120 (RTX 5080)   funnel 3.434 GH/s   plain 3.158 GH/s   +8.7%
//   sm_86  (RTX 3080)   funnel 1.903 GH/s   plain 1.920 GH/s   -0.9%
//   sm_75  (RTX 2080 Ti) funnel 1.069 GH/s  plain 1.071 GH/s   -0.2%
// The fleet is mixed, so this is chosen per architecture at compile time
// rather than hardcoded on. __CUDA_ARCH__ is set per device pass, so a
// multi-arch build still gets the right choice for each.
#ifndef HTK_USE_FUNNEL
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#    define HTK_USE_FUNNEL 1
#  else
#    define HTK_USE_FUNNEL 0
#  endif
#endif
// Pruning the final round is OFF by default because it does not pay. Measured
// GH/s, prune ON vs OFF, 2^26 nonces, best of 5, block 256:
//   sm_120 (RTX 5080)    3.382 vs 3.388   -0.2%
//   sm_89  (RTX 4090)    4.775 vs 4.994   -4.6%
//   sm_86  (RTX 3080)    1.903 vs 1.894   +0.5%
//   sm_75  (RTX 2080 Ti) 1.069 vs 1.072   -0.3%
// The op-count argument for pruning is sound -- lane 0 after round 23 needs
// only three lanes of the post-theta state -- but ptxas already dead-codes the
// unused lanes once the kernel stops STORING output lanes 1..3, so the hand
// prune buys nothing and only perturbs scheduling. Kept switchable to document
// the result. Set -DHTK_PRUNE_LAST=1 to re-enable.
#ifndef HTK_PRUNE_LAST
#define HTK_PRUNE_LAST 0
#endif
#ifndef HTK_BLOCK
#define HTK_BLOCK 256
#endif

template<int N>
__device__ __forceinline__ unsigned long long rotl64_opt(unsigned long long x) {
    static_assert(N >= 0 && N < 64, "rotation out of range");
    if (N == 0) return x;
#if HTK_USE_FUNNEL
    unsigned int lo = (unsigned int)x, hi = (unsigned int)(x >> 32), rl, rh;
    if (N == 32) { rh = lo; rl = hi; }
    else if (N < 32) {
        rh = __funnelshift_l(lo, hi, N);
        rl = __funnelshift_l(hi, lo, N);
    } else {
        constexpr int M = (N >= 32) ? (N - 32) : 0;
        rh = __funnelshift_l(hi, lo, M);
        rl = __funnelshift_l(lo, hi, M);
    }
    return ((unsigned long long)rh << 32) | (unsigned long long)rl;
#else
    return (x << N) | (x >> (64 - N));
#endif
}
#define ROTL64(N, X) rotl64_opt<(N)>(X)

// hash bytes 0..7 read big-endian == byteswap of state lane 0
__device__ __forceinline__ unsigned long long bswap64_opt(unsigned long long x) {
    unsigned int lo = (unsigned int)x, hi = (unsigned int)(x >> 32);
    return ((unsigned long long)__byte_perm(lo, 0, 0x0123) << 32) | (unsigned long long)__byte_perm(hi, 0, 0x0123);
}

#define KECCAK_ROUND(RCV) do { \
    unsigned long long c0,c1,c2,c3,c4,d0,d1,d2,d3,d4;                                                             \
    c0 = a0^a5^a10^a15^a20;                                                                             \
    c1 = a1^a6^a11^a16^a21;                                                                             \
    c2 = a2^a7^a12^a17^a22;                                                                             \
    c3 = a3^a8^a13^a18^a23;                                                                             \
    c4 = a4^a9^a14^a19^a24;                                                                             \
    d0 = c4 ^ ROTL64(1, c1);                                                                            \
    d1 = c0 ^ ROTL64(1, c2);                                                                            \
    d2 = c1 ^ ROTL64(1, c3);                                                                            \
    d3 = c2 ^ ROTL64(1, c4);                                                                            \
    d4 = c3 ^ ROTL64(1, c0);                                                                            \
    a0^=d0; a5^=d0; a10^=d0; a15^=d0; a20^=d0;                                                          \
    a1^=d1; a6^=d1; a11^=d1; a16^=d1; a21^=d1;                                                          \
    a2^=d2; a7^=d2; a12^=d2; a17^=d2; a22^=d2;                                                          \
    a3^=d3; a8^=d3; a13^=d3; a18^=d3; a23^=d3;                                                          \
    a4^=d4; a9^=d4; a14^=d4; a19^=d4; a24^=d4;                                                          \
    unsigned long long b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24; \
    b0 = a0;                                                                                            \
    b1 = ROTL64(44, a6);                                                                                \
    b2 = ROTL64(43, a12);                                                                               \
    b3 = ROTL64(21, a18);                                                                               \
    b4 = ROTL64(14, a24);                                                                               \
    b5 = ROTL64(28, a3);                                                                                \
    b6 = ROTL64(20, a9);                                                                                \
    b7 = ROTL64(3, a10);                                                                                \
    b8 = ROTL64(45, a16);                                                                               \
    b9 = ROTL64(61, a22);                                                                               \
    b10 = ROTL64(1, a1);                                                                                \
    b11 = ROTL64(6, a7);                                                                                \
    b12 = ROTL64(25, a13);                                                                              \
    b13 = ROTL64(8, a19);                                                                               \
    b14 = ROTL64(18, a20);                                                                              \
    b15 = ROTL64(27, a4);                                                                               \
    b16 = ROTL64(36, a5);                                                                               \
    b17 = ROTL64(10, a11);                                                                              \
    b18 = ROTL64(15, a17);                                                                              \
    b19 = ROTL64(56, a23);                                                                              \
    b20 = ROTL64(62, a2);                                                                               \
    b21 = ROTL64(55, a8);                                                                               \
    b22 = ROTL64(39, a14);                                                                              \
    b23 = ROTL64(41, a15);                                                                              \
    b24 = ROTL64(2, a21);                                                                               \
    a0 = b0 ^ ((~b1) & b2);                                                                             \
    a1 = b1 ^ ((~b2) & b3);                                                                             \
    a2 = b2 ^ ((~b3) & b4);                                                                             \
    a3 = b3 ^ ((~b4) & b0);                                                                             \
    a4 = b4 ^ ((~b0) & b1);                                                                             \
    a5 = b5 ^ ((~b6) & b7);                                                                             \
    a6 = b6 ^ ((~b7) & b8);                                                                             \
    a7 = b7 ^ ((~b8) & b9);                                                                             \
    a8 = b8 ^ ((~b9) & b5);                                                                             \
    a9 = b9 ^ ((~b5) & b6);                                                                             \
    a10 = b10 ^ ((~b11) & b12);                                                                         \
    a11 = b11 ^ ((~b12) & b13);                                                                         \
    a12 = b12 ^ ((~b13) & b14);                                                                         \
    a13 = b13 ^ ((~b14) & b10);                                                                         \
    a14 = b14 ^ ((~b10) & b11);                                                                         \
    a15 = b15 ^ ((~b16) & b17);                                                                         \
    a16 = b16 ^ ((~b17) & b18);                                                                         \
    a17 = b17 ^ ((~b18) & b19);                                                                         \
    a18 = b18 ^ ((~b19) & b15);                                                                         \
    a19 = b19 ^ ((~b15) & b16);                                                                         \
    a20 = b20 ^ ((~b21) & b22);                                                                         \
    a21 = b21 ^ ((~b22) & b23);                                                                         \
    a22 = b22 ^ ((~b23) & b24);                                                                         \
    a23 = b23 ^ ((~b24) & b20);                                                                         \
    a24 = b24 ^ ((~b20) & b21);                                                                         \
    a0 ^= (RCV);                                                                                        \
} while (0)

// Top 64 bits of keccak256(nonce32 || prev_hash32), big-endian -- exactly the
// value the old kernel reconstructed from hash_result[0..7]. Shared by the
// mining kernel and by the benchmark's verification kernel.
__device__ __forceinline__ unsigned long long htk_lane0_hi(unsigned long long n0, unsigned long long n1,
                                                 unsigned long long p0, unsigned long long p1,
                                                 unsigned long long p2, unsigned long long p3) {
    unsigned long long a0  = n0, a1  = n1, a2  = 0,  a3  = 0,  a4  = p0;
    unsigned long long a5  = p1, a6  = p2, a7  = p3, a8  = 0x01ULL, a9  = 0;
    unsigned long long a10 = 0,  a11 = 0,  a12 = 0,  a13 = 0,  a14 = 0;
    unsigned long long a15 = 0,  a16 = 0x8000000000000000ULL,     a17 = 0;
    unsigned long long a18 = 0,  a19 = 0,  a20 = 0,  a21 = 0,  a22 = 0;
    unsigned long long a23 = 0,  a24 = 0;

        KECCAK_ROUND(0x0000000000000001ULL);
        KECCAK_ROUND(0x0000000000008082ULL);
        KECCAK_ROUND(0x800000000000808aULL);
        KECCAK_ROUND(0x8000000080008000ULL);
        KECCAK_ROUND(0x000000000000808bULL);
        KECCAK_ROUND(0x0000000080000001ULL);
        KECCAK_ROUND(0x8000000080008081ULL);
        KECCAK_ROUND(0x8000000000008009ULL);
        KECCAK_ROUND(0x000000000000008aULL);
        KECCAK_ROUND(0x0000000000000088ULL);
        KECCAK_ROUND(0x0000000080008009ULL);
        KECCAK_ROUND(0x000000008000000aULL);
        KECCAK_ROUND(0x000000008000808bULL);
        KECCAK_ROUND(0x800000000000008bULL);
        KECCAK_ROUND(0x8000000000008089ULL);
        KECCAK_ROUND(0x8000000000008003ULL);
        KECCAK_ROUND(0x8000000000008002ULL);
        KECCAK_ROUND(0x8000000000000080ULL);
        KECCAK_ROUND(0x000000000000800aULL);
        KECCAK_ROUND(0x800000008000000aULL);
        KECCAK_ROUND(0x8000000080008081ULL);
        KECCAK_ROUND(0x8000000000008080ULL);
        KECCAK_ROUND(0x0000000080000001ULL);

#if HTK_PRUNE_LAST
    {
        unsigned long long c0,c1,c2,c3,c4,d0,d1,d2;
            c0 = a0^a5^a10^a15^a20;
            c1 = a1^a6^a11^a16^a21;
            c2 = a2^a7^a12^a17^a22;
            c3 = a3^a8^a13^a18^a23;
            c4 = a4^a9^a14^a19^a24;
            d0 = c4 ^ ROTL64(1, c1);
            d1 = c0 ^ ROTL64(1, c2);
            d2 = c1 ^ ROTL64(1, c3);
            unsigned long long t0 = a0 ^ d0, t6 = a6 ^ d1, t12 = a12 ^ d2;
            unsigned long long b0 = t0, b1 = ROTL64(44, t6), b2 = ROTL64(43, t12);
            a0 = b0 ^ ((~b1) & b2) ^ 0x8000000080008008ULL;
    }
#else
        KECCAK_ROUND(0x8000000080008008ULL);
#endif
    return bswap64_opt(a0);
}

// --------------------------------------------------------------- cold path
// Reached only when the top 64 bits already tie or beat the target (~2^-52 of
// hashes at present difficulty). Compact loop form on purpose: never hot, and
// __noinline__ keeps its registers off the search path.
__device__ const unsigned long long RC_COLD[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};
__device__ const int PILN_COLD[24] = { 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };
__device__ const int ROTC_COLD[24] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44 };

__device__ __noinline__ bool htk_exact_le(unsigned long long n0, unsigned long long n1,
                                          unsigned long long p0, unsigned long long p1,
                                          unsigned long long p2, unsigned long long p3,
                                          const unsigned char* target) {
    unsigned long long s[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) s[i] = 0;
    s[0]=n0; s[1]=n1; s[4]=p0; s[5]=p1; s[6]=p2; s[7]=p3;
    s[8] = 0x01ULL;
    s[16] ^= 0x8000000000000000ULL;
    unsigned long long C[5], temp;
    for (int round = 0; round < 24; round++) {
        for (int i = 0; i < 5; i++) C[i] = s[i]^s[i+5]^s[i+10]^s[i+15]^s[i+20];
        for (int i = 0; i < 5; i++) {
            temp = C[(i+4)%5] ^ ((C[(i+1)%5] << 1) | (C[(i+1)%5] >> 63));
            for (int j = 0; j < 25; j += 5) s[i+j] ^= temp;
        }
        temp = s[1];
        for (int i = 0; i < 24; i++) {
            int j = PILN_COLD[i];
            unsigned long long t = s[j];
            s[j] = (temp << ROTC_COLD[i]) | (temp >> (64 - ROTC_COLD[i]));
            temp = t;
        }
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) C[i] = s[j+i];
            for (int i = 0; i < 5; i++) s[j+i] ^= (~C[(i+1)%5]) & C[(i+2)%5];
        }
        s[0] ^= RC_COLD[round];
    }
    // Byte-for-byte the old kernel's comparison, equality included.
    unsigned char h[32];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned long long v = s[i];
        #pragma unroll
        for (int b = 0; b < 8; ++b) h[8*i + b] = (unsigned char)(v >> (8*b));
    }
    for (int k = 0; k < 32; ++k) {
        if (h[k] > target[k]) return false;
        if (h[k] < target[k]) return true;
    }
    return true;
}

// ------------------------------------------------------------------ kernel
// Signature is IDENTICAL to keccak_ng.cu's find_solution_kernel. Do not
// reorder, add or remove parameters: htk_miner_ng.py passes them positionally.
// NO __launch_bounds__ by default, deliberately. htk_miner_ng.py autotunes over
// BLOCK_SIZES = [128, 256, 512, 1024]; a __launch_bounds__(256) here would make
// every 512-thread launch fail and be marked UNUSABLE, silently narrowing the
// autotune relative to the kernel this replaces. Left unconstrained, register
// allocation and usable block sizes match the old kernel's freedom exactly.
// Define HTK_LAUNCH_BOUNDS to pin it. The benchmark deliberately does NOT, so
// what it measures is byte-for-byte what ships.
#ifdef HTK_LAUNCH_BOUNDS
extern "C" __global__ __launch_bounds__(HTK_LAUNCH_BOUNDS)
#else
extern "C" __global__
#endif
void HTK_KERNEL_NAME(const unsigned char* d_prev_hash,
                     const unsigned char* d_max_value_target,
                     unsigned char* d_solution_nonce,
                     unsigned long long base_nonce_part,
                     int* d_solution_found_flag,
                     unsigned long long share_target_hi,
                     unsigned long long* d_share_buf,
                     int* d_share_count,
                     int share_cap) {

    const unsigned long long thread_id =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Both buffers are cudaMalloc'd, so 8-byte aligned.
    const unsigned long long* ph = (const unsigned long long*)d_prev_hash;
    const unsigned long long p0 = ph[0], p1 = ph[1], p2 = ph[2], p3 = ph[3];
    // Top 64 bits of the 256-bit target, big-endian -- one cached load in place
    // of the old byte-at-a-time loop over global memory.
    const unsigned long long tgt_hi = bswap64_opt(((const unsigned long long*)d_max_value_target)[0]);

    const unsigned long long h_hi = htk_lane0_hi(base_nonce_part, thread_id, p0, p1, p2, p3);

    if (h_hi <= share_target_hi) {
        int idx = atomicAdd(d_share_count, 1);   // counts ALL hits, even dropped
        if (idx < share_cap) {
            d_share_buf[2 * idx]     = base_nonce_part;
            d_share_buf[2 * idx + 1] = thread_id;
        }
    }

    // h_hi < tgt_hi already decides it; h_hi == tgt_hi needs the low 192 bits.
    // Gating on <= and confirming exactly covers both without ever guessing.
    if (h_hi <= tgt_hi) {
        if (htk_exact_le(base_nonce_part, thread_id, p0, p1, p2, p3, d_max_value_target)) {
            if (atomicCAS(d_solution_found_flag, 0, 1) == 0) {
                unsigned long long* out = (unsigned long long*)d_solution_nonce;
                out[0] = base_nonce_part; out[1] = thread_id; out[2] = 0; out[3] = 0;
            }
        }
    }
}
