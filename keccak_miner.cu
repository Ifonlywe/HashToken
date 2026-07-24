#include <cuda_runtime.h>
#include <stdint.h>

#define ROL64(a, n) ( ((a) << (n)) | ((a) >> (64-(n))) )

// =======================================================================================
// === CORRECT KECCAK IMPLEMENTATION FOR SOLIDITY ===
// =======================================================================================

__device__ const uint64_t RC[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

__device__ const int keccakf_piln[24] = { 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };
__device__ const int keccakf_rotc[24] = { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44 };

__device__ void keccak_f1600(uint64_t* s) {
    uint64_t C[5], temp;
    for (int round = 0; round < 24; round++) {
        for (int i = 0; i < 5; i++) C[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
        for (int i = 0; i < 5; i++) {
            temp = C[(i + 4) % 5] ^ ROL64(C[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) s[i + j] ^= temp;
        }
        temp = s[1];
        for (int i = 0; i < 24; i++) {
            int j = keccakf_piln[i];
            uint64_t t = s[j];
            s[j] = ROL64(temp, keccakf_rotc[i]);
            temp = t;
        }
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++) C[i] = s[j + i];
            for (int i = 0; i < 5; i++) s[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
        }
        s[0] ^= RC[round];
    }
}

__device__ void keccak256_solidity(unsigned char* out, const unsigned char* in_message) {
    uint64_t s[25] = { 0 };
    const uint64_t* p = (const uint64_t*)in_message;
    s[0] = p[0]; s[1] = p[1]; s[2] = p[2]; s[3] = p[3];
    s[4] = p[4]; s[5] = p[5]; s[6] = p[6]; s[7] = p[7];
    s[8] = 0x01;
    s[16] ^= 0x8000000000000000;
    keccak_f1600(s);
    for (int i = 0; i < 4; i++) {
        ((uint64_t*)out)[i] = s[i];
    }
}

extern "C" __global__ void find_solution_kernel(const unsigned char* d_prev_hash,
                                    const unsigned char* d_max_value_target,
                                    unsigned char* d_solution_nonce,
                                    unsigned long long base_nonce_part,
                                    int* d_solution_found_flag) {

    unsigned long long thread_id = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Use shared memory for prev_hash to improve performance
    __shared__ unsigned char shared_prev_hash[32];
    if (threadIdx.x < 32) {
        shared_prev_hash[threadIdx.x] = d_prev_hash[threadIdx.x];
    }
    __syncthreads();

    // Prepare nonce and message
    unsigned char message[64];
    unsigned char current_nonce[32] = {0};

    // Set nonce: base_nonce_part + thread_id
    // This assumes base_nonce_part fills the first 8 bytes (uint64)
    // and thread_id fills the next 8 bytes (uint64)
    // The remaining 16 bytes of the 32-byte nonce are zero.
    *(unsigned long long*)(&current_nonce[0]) = base_nonce_part;
    *(unsigned long long*)(&current_nonce[8]) = thread_id;

    // Build message: nonce (32 bytes) + prev_hash (32 bytes)
    for (int i = 0; i < 32; ++i) {
        message[i] = current_nonce[i];
    }
    for (int i = 0; i < 32; ++i) {
        message[32 + i] = shared_prev_hash[i];
    }

    // Compute hash
    unsigned char hash_result[32];
    keccak256_solidity(hash_result, message);

    // Check if hash <= target (big-endian comparison)
    bool is_less_or_equal = true;
    for (int k = 0; k < 32; ++k) {
        if (hash_result[k] > d_max_value_target[k]) {
            is_less_or_equal = false;
            break;
        }
        if (hash_result[k] < d_max_value_target[k]) {
            is_less_or_equal = true;
            break;
        }
    }

    // If solution found, store it atomically
    if (is_less_or_equal) {
        // Atomic compare-and-swap: if flag is 0, set to 1 and return old value (0)
        if (atomicCAS(d_solution_found_flag, 0, 1) == 0) {
            for (int k = 0; k < 32; ++k) {
                d_solution_nonce[k] = current_nonce[k];
            }
        }
    }
}
