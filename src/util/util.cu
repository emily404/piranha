
#include "util.cuh"

#include <iostream>
#include <openssl/sha.h>
using namespace std;

extern size_t db_bytes;
extern size_t db_max_bytes;
extern size_t db_layer_max_bytes;

void log_print(std::string str) {
#if (LOG_DEBUG)
    std::cout << "----------------------------" << std::endl;
    std::cout << "Started " << str << std::endl;
    std::cout << "----------------------------" << std::endl;	
#endif
}

void error(std::string str) {
    std::cout << "Error: " << str << std::endl;
	exit(-1);
}

void printMemUsage() {

    size_t free_byte;
    size_t total_byte;
    
    auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status){
        printf("Error: cudaMemGetInfo failed with %s\n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    /*
    printf("memory usage: used = %f, free = %f, total = %f\n",
            used_db, free_db, total_db);

    printf("memory usage: used = %f, free = %f, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    */
    //printf("Allocated DeviceBuffers: %f kB\n", ((double)db_bytes)/1024.0);
    printf("Allocated DeviceBuffers: %f MB (layer max %f MB, overall max %f MB)\n", (double)db_bytes/1048576.0, (double)db_layer_max_bytes/1048576.0, (double)db_max_bytes/1048576.0);
}

// docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ uint64_t atomicAdd(uint64_t *address, uint64_t val) {

    unsigned long long int *addr_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *addr_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, val + assumed);
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return old;
}

// hash matrix
std::string sendHash[3][3] = {};
std::string verifyHash[3][3] = {};
std::string receiveHash[3][3] = {};

std::string getSendHash(int partyI, int partyJ)
{
    return sendHash[partyI][partyJ-1];
}

std::string getVerifyHash(int partyI, int partyJ)
{
    return verifyHash[partyI][partyJ-1];
}

std::string getReceiveHash(int partyI, int partyJ)
{
    return receiveHash[partyI][partyJ-1];
}

void updateSendHash(int partyI, int partyJ, std::string updated)
{
    sendHash[partyI][partyJ-1] = updated;
}

void updateVerifyHash(int partyI, int partyJ, std::string updated)
{
    verifyHash[partyI][partyJ-1] = updated;
}

void updateReceiveHash(int partyI, int partyJ, std::string updated)
{
    receiveHash[partyI][partyJ-1] = updated;
}

void printHash() {
    int rows = 3;
    int cols = 3;

    std::cout << "sendHash" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << "[" << sendHash[i][j] << "] ";
        }
        std::cout << std::endl;
    }

    std::cout << "verifyHash" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << "[" << verifyHash[i][j] << "] ";
        }
        std::cout << std::endl;
    }

    std::cout << "receiveHash" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << "[" << receiveHash[i][j] << "] ";
        }
        std::cout << std::endl;
    }

}

std::string str_sha256(const std::string &str)
{
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.c_str(), str.size());
    SHA256_Final(hash, &sha256);
    std::stringstream ss;
    for(int i = 0; i < SHA256_DIGEST_LENGTH; i++)
    {
        ss << hex << setw(2) << setfill('0') << (int)hash[i];    
    }
    
    return ss.str();
}
