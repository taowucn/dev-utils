#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMiB = 1024 * 1024;
constexpr int kBatchLaunches = 16;
volatile std::sig_atomic_t stopRequested = 0;

void handleSignal(int) { stopRequested = 1; }

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(cudaGetErrorName(status)) + ": " +
                                 cudaGetErrorString(status));
    }
}

struct Options {
    std::size_t sizeMiB = 512;
    double seconds = 3.0;
    std::string mode = "all";
    int device = 0;
};

void usage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [--mode all|read|write|copy] [--size-mib 512]"
                 " [--seconds 3] [--device 0]\n"
              << "--size-mib: size of EACH of two device buffers.\n"
              << "--seconds: wall duration PER mode; 0 runs until Ctrl-C"
                 " (select one mode).\n"
              << "Copy bandwidth counts both source reads and destination writes.\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string key = argv[argument];
        if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (argument + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        const std::string value = argv[++argument];
        std::size_t consumed = 0;
        if (key == "--mode") {
            options.mode = value;
            continue;
        }
        if (key == "--seconds") {
            options.seconds = std::stod(value, &consumed);
        } else if (key == "--device") {
            options.device = std::stoi(value, &consumed);
        } else if (key == "--size-mib") {
            if (value.empty() || value[0] == '-') {
                throw std::runtime_error("Buffer size must be positive");
            }
            const auto parsed = std::stoull(value, &consumed);
            if (parsed > std::numeric_limits<std::size_t>::max() / kMiB) {
                throw std::runtime_error("Buffer size is too large");
            }
            options.sizeMiB = static_cast<std::size_t>(parsed);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
        if (consumed != value.size()) {
            throw std::runtime_error("Invalid value for " + key);
        }
    }
    if (options.sizeMiB == 0 || options.device < 0 ||
        !std::isfinite(options.seconds) || options.seconds < 0) {
        throw std::runtime_error("Invalid size, device, or duration");
    }
    if (options.mode != "all" && options.mode != "read" &&
        options.mode != "write" && options.mode != "copy") {
        throw std::runtime_error("Mode must be all, read, write, or copy");
    }
    if (options.seconds == 0 && options.mode == "all") {
        throw std::runtime_error("Continuous operation requires a single mode");
    }
    return options;
}

__device__ uint4 pattern(std::size_t index) {
    const auto value = static_cast<unsigned int>(index);
    return make_uint4(value, value ^ 0x9e3779b9U, value + 17U, ~value);
}

__device__ __forceinline__ uint4 loadVector(const uint4* address) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void storeVector(uint4* address, uint4 value) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 : : "l"(address), "r"(value.x), "r"(value.y),
                     "r"(value.z), "r"(value.w) : "memory");
}

template <int Mode>
__global__ void bandwidthKernel(const uint4* __restrict__ source,
                                uint4* __restrict__ destination,
                                uint4* __restrict__ checksum, std::size_t count) {
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    uint4 accumulator = make_uint4(0, 0, 0, 0);
    for (std::size_t index = thread; index < count; index += stride) {
        if constexpr (Mode == 0) {
            const uint4 value = loadVector(source + index);
            accumulator.x ^= value.x;
            accumulator.y ^= value.y;
            accumulator.z ^= value.z;
            accumulator.w ^= value.w;
        } else if constexpr (Mode == 1) {
            storeVector(destination + index, pattern(index));
        } else {
            storeVector(destination + index, loadVector(source + index));
        }
    }
    if constexpr (Mode == 0) {
        checksum[thread] = accumulator;
    }
}

__global__ void validateKernel(const uint4* data, std::size_t count,
                               unsigned int* errors) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const uint4 actual = data[index];
        const uint4 expected = pattern(index);
        if (actual.x != expected.x || actual.y != expected.y ||
            actual.z != expected.z || actual.w != expected.w) {
            atomicExch(errors, 1U);
        }
    }
}

struct Resources {
    uint4* source = nullptr;
    uint4* destination = nullptr;
    uint4* checksum = nullptr;
    unsigned int* errors = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;

    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (end) cudaEventDestroy(end);
        if (begin) cudaEventDestroy(begin);
        if (errors) cudaFree(errors);
        if (checksum) cudaFree(checksum);
        if (destination) cudaFree(destination);
        if (source) cudaFree(source);
        if (stream) cudaStreamDestroy(stream);
    }
};

struct Launch {
    int blocks;
    int threads;
};

void launch(const std::string& mode, const Launch& config, Resources& resources,
            std::size_t count) {
    if (mode == "read") {
        bandwidthKernel<0><<<config.blocks, config.threads, 0, resources.stream>>>(
            resources.source, resources.destination, resources.checksum, count);
    } else if (mode == "write") {
        bandwidthKernel<1><<<config.blocks, config.threads, 0, resources.stream>>>(
            resources.source, resources.destination, resources.checksum, count);
    } else {
        bandwidthKernel<2><<<config.blocks, config.threads, 0, resources.stream>>>(
            resources.source, resources.destination, resources.checksum, count);
    }
    check(cudaGetLastError());
}

float measure(const std::string& mode, const Launch& config, Resources& resources,
              std::size_t count, int repetitions) {
    check(cudaEventRecord(resources.begin, resources.stream));
    for (int iteration = 0; iteration < repetitions; ++iteration) {
        launch(mode, config, resources, count);
    }
    check(cudaEventRecord(resources.end, resources.stream));
    check(cudaEventSynchronize(resources.end));
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, resources.begin, resources.end));
    if (milliseconds <= 0) throw std::runtime_error("Invalid CUDA event duration");
    return milliseconds;
}

void validate(const std::string& mode, const Launch& config, Resources& resources,
              std::size_t count) {
    if (mode == "read") {
        const std::size_t threads = static_cast<std::size_t>(config.blocks) * config.threads;
        std::vector<uint4> actual(threads);
        check(cudaMemcpyAsync(actual.data(), resources.checksum, threads * sizeof(uint4),
                              cudaMemcpyDeviceToHost, resources.stream));
        check(cudaStreamSynchronize(resources.stream));
        for (std::size_t thread = 0; thread < threads; ++thread) {
            uint4 expected = make_uint4(0, 0, 0, 0);
            for (std::size_t index = thread; index < count; index += threads) {
                const auto value = static_cast<unsigned int>(index);
                expected.x ^= value;
                expected.y ^= value ^ 0x9e3779b9U;
                expected.z ^= value + 17U;
                expected.w ^= ~value;
            }
            if (actual[thread].x != expected.x || actual[thread].y != expected.y ||
                actual[thread].z != expected.z || actual[thread].w != expected.w) {
                throw std::runtime_error("Read checksum validation failed");
            }
        }
    } else {
        check(cudaMemsetAsync(resources.errors, 0, sizeof(unsigned int), resources.stream));
        validateKernel<<<config.blocks, config.threads, 0, resources.stream>>>(
            resources.destination, count, resources.errors);
        check(cudaGetLastError());
        unsigned int errors = 0;
        check(cudaMemcpyAsync(&errors, resources.errors, sizeof(errors),
                              cudaMemcpyDeviceToHost, resources.stream));
        check(cudaStreamSynchronize(resources.stream));
        if (errors) throw std::runtime_error("Write/copy data validation failed");
    }
}

void benchmark(const std::string& mode, const Options& options,
               const cudaDeviceProp& properties, Resources& resources,
               std::size_t bytes) {
    const std::size_t count = bytes / sizeof(uint4);
    Launch best{properties.multiProcessorCount * 4, 256};
    float bestTime = std::numeric_limits<float>::max();
    for (int threads : {128, 256, 512}) {
        if (threads > properties.maxThreadsPerBlock) continue;
        for (int blocksPerSM : {4, 8, 16, 32}) {
            if (stopRequested) return;
            const Launch candidate{properties.multiProcessorCount * blocksPerSM, threads};
            launch(mode, candidate, resources, count);
            const float elapsed = measure(mode, candidate, resources, count, 3);
            if (elapsed < bestTime) {
                bestTime = elapsed;
                best = candidate;
            }
        }
    }
    const double transferBytes = static_cast<double>(bytes) * (mode == "copy" ? 2 : 1);
    std::cout << "\n" << mode << ": blocks=" << best.blocks << ", threads=" << best.threads
              << ", tuning best=" << transferBytes * 3 / (bestTime * 1e6) << " GB/s\n";
    const auto start = std::chrono::steady_clock::now();
    auto lastReport = start;
    double totalMilliseconds = 0;
    double peakGBs = 0;
    std::uint64_t launches = 0;
    while (!stopRequested) {
        const float elapsed = measure(mode, best, resources, count, kBatchLaunches);
        totalMilliseconds += elapsed;
        launches += kBatchLaunches;
        peakGBs = std::max(peakGBs, transferBytes * kBatchLaunches / (elapsed * 1e6));
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - lastReport).count() >= 1) {
            std::cout << "  elapsed=" << std::chrono::duration<double>(now - start).count()
                      << " s, event-average=" << transferBytes * launches / (totalMilliseconds * 1e6)
                      << " GB/s" << std::endl;
            lastReport = now;
        }
        if (options.seconds > 0 &&
            std::chrono::duration<double>(now - start).count() >= options.seconds) break;
    }
    const double wallSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (launches == 0) return;
    std::cout << "  GPU-event average: " << transferBytes * launches / (totalMilliseconds * 1e6)
              << " GB/s\n  Wall-time average: " << transferBytes * launches / (wallSeconds * 1e9)
              << " GB/s\n  Best batch: " << peakGBs << " GB/s\n";
    validate(mode, best, resources, count);
    std::cout << "  Data validation: PASS\n";
}

}

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        std::signal(SIGINT, handleSignal);
        std::signal(SIGTERM, handleSignal);
        check(cudaSetDevice(options.device));
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, options.device));
        const std::size_t bytes = options.sizeMiB * kMiB;
        if (properties.l2CacheSize > 0 &&
            bytes / 4 < static_cast<std::size_t>(properties.l2CacheSize)) {
            throw std::runtime_error("Each buffer must be at least 4 times the GPU L2 size");
        }
        const std::size_t checksumBytes = static_cast<std::size_t>(properties.multiProcessorCount)
                                         * 32 * 512 * sizeof(uint4);
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        check(cudaMemGetInfo(&freeBytes, &totalBytes));
        if (checksumBytes + sizeof(unsigned int) > freeBytes / 5 ||
            bytes > (freeBytes - freeBytes / 5 - checksumBytes - sizeof(unsigned int)) / 2) {
            throw std::runtime_error("Insufficient free GPU memory with 20% headroom; reduce --size-mib");
        }
        Resources resources;
        check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
        check(cudaEventCreate(&resources.begin));
        check(cudaEventCreate(&resources.end));
        check(cudaMalloc(&resources.source, bytes));
        check(cudaMalloc(&resources.destination, bytes));
        check(cudaMalloc(&resources.checksum, checksumBytes));
        check(cudaMalloc(&resources.errors, sizeof(unsigned int)));
        const Launch initialization{properties.multiProcessorCount * 8, 256};
        bandwidthKernel<1><<<initialization.blocks, initialization.threads, 0, resources.stream>>>(
            nullptr, resources.source, nullptr, bytes / sizeof(uint4));
        check(cudaGetLastError());
        check(cudaMemsetAsync(resources.destination, 0, bytes, resources.stream));
        check(cudaStreamSynchronize(resources.stream));
        std::cout << std::fixed << std::setprecision(2)
                  << "GPU: " << properties.name << ", SMs: " << properties.multiProcessorCount
                  << ", L2: " << properties.l2CacheSize / static_cast<double>(kMiB) << " MiB\n"
                  << "Two buffers: " << options.sizeMiB << " MiB each; 128-bit coalesced accesses.\n"
                  << "Loads bypass L1 (.cg); stores use streaming policy (.cs).\n"
                  << "Logical GB/s, not hardware DRAM byte counters. Copy counts read + write.\n"
                  << "Observe actual EMC utilization with: sudo tegrastats --interval 100\n"
                  << "No clock or power settings are modified. Ctrl-C stops after the current batch.\n";
        for (const std::string& mode : std::vector<std::string>{"read", "write", "copy"}) {
            if (stopRequested) break;
            if (options.mode == "all" || options.mode == mode) {
                benchmark(mode, options, properties, resources, bytes);
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}