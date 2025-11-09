// author: https://t.me/biernus
#include "secp256k1.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <fstream>
#include <stdint.h>
#include <curand_kernel.h>
#include <algorithm>
#include <random>
#include <inttypes.h>
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <chrono>
#pragma once

__device__ __host__ __forceinline__ uint8_t hex_char_to_byte(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

// Convert hex string to bytes
__device__ __host__ __device__ void hex_string_to_bytes(const char* hex_str, uint8_t* bytes, int num_bytes) {
    #pragma unroll 8
    for (int i = 0; i < num_bytes; i++) {
        bytes[i] = (hex_char_to_byte(hex_str[i * 2]) << 4) | 
                   hex_char_to_byte(hex_str[i * 2 + 1]);
    }
}


// Convert hex string to BigInt - optimized
__device__ __host__ void hex_to_bigint(const char* hex_str, BigInt* bigint) {
    // Initialize all data to 0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bigint->data[i] = 0;
    }
    
    int len = 0;
    while (hex_str[len] != '\0' && len < 64) len++;
    
    // Process hex string from right to left
    int word_idx = 0;
    int bit_offset = 0;
    
    for (int i = len - 1; i >= 0 && word_idx < 8; i--) {
        uint8_t val = hex_char_to_byte(hex_str[i]);
        
        bigint->data[word_idx] |= ((uint32_t)val << bit_offset);
        
        bit_offset += 4;
        if (bit_offset >= 32) {
            bit_offset = 0;
            word_idx++;
        }
    }
}

// Convert BigInt to hex string - optimized
__device__ void bigint_to_hex(const BigInt* bigint, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    int idx = 0;
    bool leading_zero = true;
    
    // Process from most significant word to least
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        for (int j = 28; j >= 0; j -= 4) {
            uint8_t nibble = (bigint->data[i] >> j) & 0xF;
            if (nibble != 0 || !leading_zero || (i == 0 && j == 0)) {
                hex_str[idx++] = hex_chars[nibble];
                leading_zero = false;
            }
        }
    }
    
    // Handle case where number is 0
    if (idx == 0) {
        hex_str[idx++] = '0';
    }
    
    hex_str[idx] = '\0';
}

// Optimized byte to hex conversion
__device__ __forceinline__ void byte_to_hex(uint8_t byte, char* out) {
    const char hex_chars[] = "0123456789abcdef";
    out[0] = hex_chars[(byte >> 4) & 0xF];
    out[1] = hex_chars[byte & 0xF];
}

__device__ void hash160_to_hex(uint8_t* hash, char* hex_str) {
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        byte_to_hex(hash[i], &hex_str[i * 2]);
    }
    hex_str[40] = '\0';
}


__device__ __forceinline__ bool compare_hash160_fast(const uint8_t* hash1, const uint8_t* hash2) {
    uint64_t a1, a2, b1, b2;
    uint32_t c1, c2;
    
    memcpy(&a1, hash1, 8);
    memcpy(&a2, hash1 + 8, 8);
    memcpy(&c1, hash1 + 16, 4);

    memcpy(&b1, hash2, 8);
    memcpy(&b2, hash2 + 8, 8);
    memcpy(&c2, hash2 + 16, 4);

    return (a1 == b1) && (a2 == b2) && (c1 == c2);
}

__device__ void hash160_to_hex(const uint8_t *hash, char *out_hex) {
    const char hex_chars[] = "0123456789abcdef";
    for (int i = 0; i < 20; ++i) {
        out_hex[i * 2]     = hex_chars[hash[i] >> 4];
        out_hex[i * 2 + 1] = hex_chars[hash[i] & 0x0F];
    }
    out_hex[40] = '\0';
}

__device__ int bigint_compare(const BigInt* a, const BigInt* b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; --i) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__device__ void bigint_copy(struct BigInt *dest, const struct BigInt *src) {
    for (int i = 0; i < BIGINT_WORDS; i++) {
        dest->data[i] = src->data[i];
    }
}
// Global device constants for min/max as BigInt
__constant__ BigInt d_min_bigint;
__constant__ BigInt d_max_bigint;

__device__ volatile int g_found = 0;
__device__ char g_found_hex[65] = {0};
__device__ char g_found_hash160[41] = {0};

__device__ char d_min_hex[65];
__device__ char d_max_hex[65];
__device__ int d_hex_length;
__device__ int d_prefix_length;  // Length of common prefix

// Fisher-Yates shuffle on device
__device__ void shuffle_hex_pool(char* pool, int length, curandStatePhilox4_32_10_t* state) {
    for (int i = length - 1; i > 0; i--) {
        int j = curand(state) % (i + 1);
        char temp = pool[i];
        pool[i] = pool[j];
        pool[j] = temp;
    }
}

// Extract hex chars from pool and build a key within range
__device__ void build_key_from_pool(BigInt* key, const char* prefix, int prefix_len, 
                                     const char* pool, int pool_size, int offset,
                                     curandStatePhilox4_32_10_t* state) {
    char hex_str[65] = {0};
    
    // Copy prefix
    for (int i = 0; i < prefix_len; i++) {
        hex_str[i] = prefix[i];
    }
    
    // Fill remaining from pool with offset
    int remaining = d_hex_length - prefix_len;
    for (int i = 0; i < remaining; i++) {
        int pool_idx = (offset + i) % pool_size;
        hex_str[prefix_len + i] = pool[pool_idx];
    }
    
    // Convert to BigInt
    hex_to_bigint(hex_str, key);
    
    // Clamp to range [min, max]
    if (bigint_compare(key, &d_min_bigint) < 0) {
        bigint_copy(key, &d_min_bigint);
    } else if (bigint_compare(key, &d_max_bigint) > 0) {
        bigint_copy(key, &d_max_bigint);
    }
}

__global__ void start(const uint8_t* target, uint64_t seed, int total_threads, const char* hex_pool, int pool_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize RNG
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    
    // Batch storage
    ECPointJac result_jac_batch[BATCH_SIZE];
    BigInt priv_batch[BATCH_SIZE];
    uint8_t hash160_batch[BATCH_SIZE][20];
    
    // Each batch iteration uses different shuffle
    #pragma unroll
    for (int i = 0; i < BATCH_SIZE; ++i) {
        // Allocate working copy of a portion of the pool
        char batch_pool[512];  // Local buffer for shuffling
        int copy_size = (512 < pool_size) ? 512 : pool_size;
        
        // Copy a portion of the pool starting from thread-specific offset
        int start_offset = (tid * 512 + i * 1024) % pool_size;
        for (int j = 0; j < copy_size; j++) {
            batch_pool[j] = hex_pool[(start_offset + j) % pool_size];
        }
        
        // Shuffle the local pool for this batch item
        shuffle_hex_pool(batch_pool, copy_size, &state);
        
        // Build key from shuffled pool with thread offset
        build_key_from_pool(&priv_batch[i], d_min_hex, d_prefix_length, 
                           batch_pool, copy_size, tid + i * total_threads, &state);
        
        // Compute public key
        scalar_multiply_multi_base_jac(&result_jac_batch[i], &priv_batch[i]);
    }
    
    // Batch convert to hash160
    jacobian_batch_to_hash160(result_jac_batch, hash160_batch);
    
    // Check results
    #pragma unroll
    for (int i = 0; i < BATCH_SIZE; ++i) {
        if (compare_hash160_fast(hash160_batch[i], target)) {
            if (atomicCAS((int*)&g_found, 0, 1) == 0) {
                bigint_to_hex(&priv_batch[i], g_found_hex);
                hash160_to_hex(hash160_batch[i], g_found_hash160);
            }
            return;
        }
    }
}

// Generate hex pool with all possible hex characters
void generate_hex_pool(char* pool, size_t size) {
    const char hex_chars[] = "0123456789abcdef";
    for (size_t i = 0; i < size; i++) {
        pool[i] = hex_chars[rand() % 16];
    }
}

// Shuffle hex pool on host (Fisher-Yates)
void shuffle_hex_pool_host(char* pool, size_t size) {
    for (size_t i = size - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        char temp = pool[i];
        pool[i] = pool[j];
        pool[j] = temp;
    }
}

// Find common prefix between min and max
int find_common_prefix(const char* min, const char* max) {
    int len = 0;
    while (min[len] && max[len] && min[len] == max[len]) {
        len++;
    }
    return len;
}

bool run_with_quantum_data(const char* min, const char* max, const char* target, int blocks, int threads, int device_id) {
    uint8_t shared_target[20];
    hex_string_to_bytes(target, shared_target, 20);
    uint8_t *d_target;
    cudaMalloc(&d_target, 20);
    cudaMemcpy(d_target, shared_target, 20, cudaMemcpyHostToDevice);
    
    // Convert min and max hex strings to BigInt and copy to device
    BigInt min_bigint, max_bigint;
    hex_to_bigint(min, &min_bigint);
    hex_to_bigint(max, &max_bigint);
    
    cudaMemcpyToSymbol(d_min_bigint, &min_bigint, sizeof(BigInt));
    cudaMemcpyToSymbol(d_max_bigint, &max_bigint, sizeof(BigInt));
    
    // Find common prefix
    int prefix_length = find_common_prefix(min, max);
    int hex_length = strlen(min);
    
    cudaMemcpyToSymbol(d_hex_length, &hex_length, sizeof(int));
    cudaMemcpyToSymbol(d_prefix_length, &prefix_length, sizeof(int));
    cudaMemcpyToSymbol(d_min_hex, min, hex_length + 1);
    cudaMemcpyToSymbol(d_max_hex, max, hex_length + 1);
    
    int total_threads = blocks * threads;
    
    // Calculate optimal pool size: 
    // We want each thread to have unique starting positions
    // Pool size = total_threads * (hex_length - prefix_length) * BATCH_SIZE
    // But cap it at reasonable size for memory (e.g., 100M characters = ~100MB)
    int chars_per_key = hex_length - prefix_length;
    size_t ideal_pool_size = (size_t)total_threads * chars_per_key * BATCH_SIZE;
    size_t max_pool_size = 100 * 1024 * 1024; // 100MB max
    size_t pool_size = (ideal_pool_size < max_pool_size) ? ideal_pool_size : max_pool_size;
    
    // Ensure pool_size is at least 10x the number of threads for good distribution
    size_t min_pool_size = total_threads * 10;
    if (pool_size < min_pool_size) pool_size = min_pool_size;
    
    printf("Pool size calculation:\n");
    printf("  Total threads: %d\n", total_threads);
    printf("  Chars per key: %d\n", chars_per_key);
    printf("  Ideal pool size: %zu (%.2f MB)\n", ideal_pool_size, ideal_pool_size / (1024.0 * 1024.0));
    printf("  Actual pool size: %zu (%.2f MB)\n\n", pool_size, pool_size / (1024.0 * 1024.0));
    
    // Generate initial hex pool
    char* hex_pool = new char[pool_size];
    srand(time(NULL));
    generate_hex_pool(hex_pool, pool_size);
    
    // Allocate device memory for hex pool
    char* d_hex_pool;
    cudaMalloc(&d_hex_pool, pool_size);
    
    int found_flag;
    
    // Calculate keys processed per kernel launch
    uint64_t keys_per_kernel = (uint64_t)blocks * threads * BATCH_SIZE;
    
    printf("Searching in range:\n");
    printf("Min: %s\n", min);
    printf("Max: %s\n", max);
    printf("Target: %s\n", target);
    printf("Common prefix length: %d\n", prefix_length);
    printf("Hex length: %d\n", hex_length);
    printf("Blocks: %d, Threads: %d, Batch size: %d\n", blocks, threads, BATCH_SIZE);
    printf("Total threads: %d\n", total_threads);
    printf("Keys per kernel: %llu\n\n", (unsigned long long)keys_per_kernel);
    
    // Performance tracking variables
    uint64_t total_keys_checked = 0;
    uint64_t kernel_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print_time = start_time;
    
    while(true) {
        // Shuffle the hex pool for this kernel launch
        shuffle_hex_pool_host(hex_pool, pool_size);
        cudaMemcpy(d_hex_pool, hex_pool, pool_size, cudaMemcpyHostToDevice);
        
        // Generate random seed for this kernel
        uint64_t seed = ((uint64_t)rand() << 32) | rand();
        
        auto kernel_start = std::chrono::high_resolution_clock::now();
        
        // Launch kernel
        start<<<blocks, threads>>>(d_target, seed, total_threads, d_hex_pool, (int)pool_size);
        cudaDeviceSynchronize();
        
        auto kernel_end = std::chrono::high_resolution_clock::now();
        
        // Calculate kernel execution time
        double kernel_time = std::chrono::duration<double>(kernel_end - kernel_start).count();
        
        // Update counters
        total_keys_checked += keys_per_kernel;
        kernel_count++;
        
        // Print performance stats every second
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed_since_print = std::chrono::duration<double>(current_time - last_print_time).count();
        
        if (elapsed_since_print >= 1.0) {
            double current_kps = keys_per_kernel / kernel_time;
            
            printf("\rKernels: %llu | Speed: %.2f MK/s | Total: %.2f B keys",
                   (unsigned long long)kernel_count,
                   current_kps / 1000000.0,
                   total_keys_checked / 1000000000.0);
            fflush(stdout);
            
            last_print_time = current_time;
        }
        
        // Check if key was found
        cudaMemcpyFromSymbol(&found_flag, g_found, sizeof(int));
        if (found_flag) {
            printf("\n\n");
            
            char found_hex[65], found_hash160[41];
            cudaMemcpyFromSymbol(found_hex, g_found_hex, 65);
            cudaMemcpyFromSymbol(found_hash160, g_found_hash160, 41);
            
            double total_time = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start_time
            ).count();
            
            printf("FOUND!\n");
            printf("Private Key: %s\n", found_hex);
            printf("Hash160: %s\n", found_hash160);
            printf("Total time: %.2f seconds\n", total_time);
            printf("Total keys checked: %llu (%.2f billion)\n", 
                   (unsigned long long)total_keys_checked,
                   total_keys_checked / 1000000000.0);
            printf("Average speed: %.2f MK/s\n", total_keys_checked / total_time / 1000000.0);
            
            std::ofstream outfile("result.txt", std::ios::app);
            if (outfile.is_open()) {
                std::time_t now = std::time(nullptr);
                char timestamp[100];
                std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
                outfile << "[" << timestamp << "] Found: " << found_hex << " -> " << found_hash160 << std::endl;
                outfile << "Total keys checked: " << total_keys_checked << std::endl;
                outfile << "Time taken: " << total_time << " seconds" << std::endl;
                outfile << "Average speed: " << (total_keys_checked / total_time / 1000000.0) << " MK/s" << std::endl;
                outfile << std::endl;
                outfile.close();
                std::cout << "Result appended to result.txt" << std::endl;
            }
            
            delete[] hex_pool;
            cudaFree(d_hex_pool);
            cudaFree(d_target);
            return true;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <min> <max> <target> [device_id]" << std::endl;
        return 1;
    }
    int blocks = 4096;
    int threads = 256;
    int device_id = (argc > 4) ? std::stoi(argv[4]) : 0;
    
    // Set GPU device
    cudaSetDevice(device_id);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error setting device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Validate input lengths match
    if (strlen(argv[1]) != strlen(argv[2])) {
        std::cerr << "Error: min and max must have the same length" << std::endl;
        return 1;
    }
    init_gpu_constants();
    cudaDeviceSynchronize();
    bool result = run_with_quantum_data(argv[1], argv[2], argv[3], blocks, threads, device_id);
    
    return 0;
}