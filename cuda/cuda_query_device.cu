#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr double kBytesPerKiB = 1024.0;
constexpr double kBytesPerMiB = kBytesPerKiB * 1024.0;
constexpr double kBytesPerGiB = kBytesPerMiB * 1024.0;

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorName(status) << " ("
              << cudaGetErrorString(status) << ")\n";
    return false;
}

template <typename Value>
void printField(const char* label, const Value& value) {
    std::cout << "  " << std::left << std::setw(42) << label << value << '\n';
}

void printBytes(const char* label, std::size_t bytes) {
    std::ostringstream text;
    text << bytes << " bytes (" << std::fixed << std::setprecision(2);
    if (bytes >= kBytesPerGiB) {
        text << bytes / kBytesPerGiB << " GiB)";
    } else if (bytes >= kBytesPerMiB) {
        text << bytes / kBytesPerMiB << " MiB)";
    } else {
        text << bytes / kBytesPerKiB << " KiB)";
    }
    printField(label, text.str());
}

void printFlag(const char* label, int enabled) {
    printField(label, enabled ? "yes" : "no");
}

std::string versionString(int version) {
    return std::to_string(version / 1000) + "." +
           std::to_string((version % 1000) / 10);
}

bool printDevice(int deviceIndex) {
    cudaDeviceProp properties{};
    if (!checkCuda(cudaGetDeviceProperties(&properties, deviceIndex),
                   "cudaGetDeviceProperties") ||
        !checkCuda(cudaSetDevice(deviceIndex), "cudaSetDevice")) {
        return false;
    }

    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    if (!checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo")) {
        return false;
    }

    std::cout << "\n=== Device " << deviceIndex << ": " << properties.name
              << " ===\n\nIdentity\n";
    printField("Compute capability", std::to_string(properties.major) + "." +
                                          std::to_string(properties.minor));
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (int byteIndex = 0; byteIndex < 16; ++byteIndex) {
        if (byteIndex == 4 || byteIndex == 6 || byteIndex == 8 || byteIndex == 10) {
            uuid << '-';
        }
        uuid << std::setw(2)
             << static_cast<unsigned int>(
                    static_cast<unsigned char>(properties.uuid.bytes[byteIndex]));
    }
    printField("Device UUID", uuid.str());
    printFlag("Integrated GPU", properties.integrated);
    printFlag("ECC enabled", properties.ECCEnabled);

    std::cout << "\nSM resources and execution limits\n";
    printField("SM count", properties.multiProcessorCount);
    printField("Threads per warp", properties.warpSize);
    printField("Max resident threads per SM", properties.maxThreadsPerMultiProcessor);
    printField("Max resident warps per SM",
               properties.maxThreadsPerMultiProcessor / properties.warpSize);
    printField("Max resident blocks per SM", properties.maxBlocksPerMultiProcessor);
    printField("32-bit registers per SM", properties.regsPerMultiprocessor);
    printBytes("Register capacity per SM",
               static_cast<std::size_t>(properties.regsPerMultiprocessor) * 4);
    printBytes("Shared memory per SM", properties.sharedMemPerMultiprocessor);
    printField("Max threads per block", properties.maxThreadsPerBlock);
    printField("32-bit registers per block", properties.regsPerBlock);
    printBytes("Shared memory per block (default)", properties.sharedMemPerBlock);
    printBytes("Shared memory per block (opt-in)", properties.sharedMemPerBlockOptin);
    printBytes("Driver-reserved shared memory per block",
               properties.reservedSharedMemPerBlock);
    printField("Max block dimensions (x, y, z)",
               std::to_string(properties.maxThreadsDim[0]) + ", " +
                   std::to_string(properties.maxThreadsDim[1]) + ", " +
                   std::to_string(properties.maxThreadsDim[2]));
    printField("Max grid dimensions (x, y, z)",
               std::to_string(properties.maxGridSize[0]) + ", " +
                   std::to_string(properties.maxGridSize[1]) + ", " +
                   std::to_string(properties.maxGridSize[2]));

    std::cout << "\nMemory and cache\n";
    printBytes("Global memory (device property)", properties.totalGlobalMem);
    printBytes("Memory total (runtime snapshot)", totalBytes);
    printBytes("Memory free (runtime snapshot)", freeBytes);
    printBytes("L2 cache (whole GPU)", properties.l2CacheSize);
    printBytes("Max persisting L2 cache", properties.persistingL2CacheMaxSize);
    printBytes("Max access-policy window", properties.accessPolicyMaxWindowSize);
    printBytes("Constant memory", properties.totalConstMem);
    printField("Memory bus width (bits)", properties.memoryBusWidth);
    printBytes("Maximum memory pitch", properties.memPitch);
    printBytes("Texture alignment", properties.textureAlignment);
    printFlag("Global memory L1 caching", properties.globalL1CacheSupported);
    printFlag("Local memory L1 caching", properties.localL1CacheSupported);

    std::cout << "\nCapabilities\n";
    printFlag("Unified virtual addressing", properties.unifiedAddressing);
    printFlag("Managed memory", properties.managedMemory);
    printFlag("Concurrent managed access", properties.concurrentManagedAccess);
    printFlag("Pageable memory access", properties.pageableMemoryAccess);
    printFlag("Pageable access uses host page tables",
              properties.pageableMemoryAccessUsesHostPageTables);
    printFlag("Direct managed access from host", properties.directManagedMemAccessFromHost);
    printFlag("Map host memory", properties.canMapHostMemory);
    printFlag("Host native atomics", properties.hostNativeAtomicSupported);
    printFlag("Concurrent kernels", properties.concurrentKernels);
    printField("Asynchronous engine count", properties.asyncEngineCount);
    printFlag("Stream priorities", properties.streamPrioritiesSupported);
    printFlag("Cooperative launch", properties.cooperativeLaunch);
    printFlag("Thread-block cluster launch", properties.clusterLaunch);
    printFlag("Memory pools", properties.memoryPoolsSupported);

    std::cout << "\nSM limits are hardware ceilings, not measured occupancy.\n"
              << "CUDA core count, Tensor Core count, and L1 capacity are not "
                 "reported by cudaGetDeviceProperties().\n";
    return true;
}

}  // namespace

int main() {
    int driverVersion = 0;
    int runtimeVersion = 0;
    int deviceCount = 0;
    if (!checkCuda(cudaDriverGetVersion(&driverVersion), "cudaDriverGetVersion") ||
        !checkCuda(cudaRuntimeGetVersion(&runtimeVersion), "cudaRuntimeGetVersion") ||
        !checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) {
        return EXIT_FAILURE;
    }

    printField("CUDA driver API version", versionString(driverVersion));
    printField("CUDA runtime version", versionString(runtimeVersion));
    printField("CUDA device count", deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices are visible.\n";
        return EXIT_FAILURE;
    }

    bool success = true;
    for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
        if (!printDevice(deviceIndex)) {
            success = false;
        }
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}