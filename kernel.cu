
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

#include <cstdio>

constexpr int width = 10;
constexpr int height = 10;
constexpr int size = width * height;

constexpr int TILE_W = 14;
constexpr int TILE_H = 14;

bool cudaLogErrorInternal(cudaError_t error, const char* file, uint32_t line, const char* function,
    const char* cmd) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Cuda Error %s:%u %s\n\t%s\nReturned error:\n\t%s:\t%s", file, line, function, cmd,
            cudaGetErrorName(error), cudaGetErrorString(error));
        return false;
    }
    return true;
}

#define cudaLogError(cmd) cudaLogErrorInternal((cmd), __FILE__, __LINE__, __PRETTY_FUNCTION__, #cmd)
#define cudaLogFatalError(cmd)                                                                                         \
  do {                                                                                                                 \
    if (!cudaLogErrorInternal((cmd), __FILE__, __LINE__, __PRETTY_FUNCTION__, #cmd))                                   \
      abort();                                                                                                         \
  } while (0)

__device__ int getIdx(int x, int y) {
    y += (y < 0) ? height : (y >= height) ? -height : 0;
    x += (x < 0) ? width : (x >= width) ? -width : 0;
    int i = (y ) * width + (x );
    return i;
}

__global__ void stepKernel(const uint8_t* state, uint8_t* out)
{
    int x = TILE_W * (blockIdx.x - 1) + (threadIdx.x - 1);
    int y = TILE_H * (blockIdx.y - 1) + (threadIdx.y - 1);

    __shared__ uint8_t tile[TILE_H + 2][TILE_W + 2];

    int C = getIdx(x, y);
    tile[threadIdx.y][threadIdx.x] = state[C];

    if (x >= width || y >= height || threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x > TILE_W || threadIdx.y > TILE_H) {
        return;
    }
    
    __syncthreads();

    x = threadIdx.x;
    y = threadIdx.y;
    int N = tile[y - 1][x];
    int NE = tile[y - 1][x + 1];
    int NW = tile[y - 1][x - 1];
    int E = tile[y][x + 1];
    int W = tile[y][x - 1];
    int S = tile[y + 1][x];
    int SE = tile[y + 1][x + 1];
    int SW = tile[y + 1][x - 1];

    int total = N + NE + NW + E + W + S + SE + SW;
    out[C] = tile[y][x] && total == 2 || total == 3;
}

cudaError_t stepWithCuda(const std::vector<uint8_t>& a, std::vector<uint8_t>& b, uint8_t* dev_a, uint8_t* dev_b, cudaStream_t stream)
{
    cudaError_t cudaStatus;

    dim3 threadsPerBlock(TILE_W + 2, TILE_H + 2, 1);

    dim3 blocksPerGrid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H, 1);
    stepKernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (dev_a, dev_b);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    cudaStatus = cudaMemcpyAsync(b.data(), dev_b, size * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync failed!");
        return cudaStatus;
    }

    return cudaStatus;
}

void draw(std::vector<uint8_t>& state) {
    for (int i = 0; i < width * height; ++i) {
        if (state[i]) { std::cout << "*"; }
        else { std::cout << "."; }

        if (i % width == width - 1) { std::cout << "\n"; }

    }
    for (int i = 0; i < 10; ++i) {
        std::cout << "\n";
    }
}

void app(std::vector<uint8_t>& state, std::vector<uint8_t>& out) {

    uint8_t* dev_a = 0; //should point to block of gpu memory
    uint8_t* dev_b = 0;


    cudaStream_t stream{};
    cudaLogFatalError(cudaStreamCreate(&stream));
    cudaLogFatalError(cudaMalloc((void**)&dev_a, size * sizeof(uint8_t)));
    cudaLogFatalError(cudaMalloc((void**)&dev_b, size * sizeof(uint8_t)));

    //async allows you to choose what stream, and returns immediately
    cudaLogFatalError(cudaMemcpyAsync(dev_a, state.data(), size * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

    cudaLogFatalError(stepWithCuda(state, out, dev_a, dev_b, stream));

    draw(state);
    while (1) {
        char key = 'x';
        // std::cin >> key;
        if (key == 'x')
        {
            stepWithCuda(state, out, dev_a, dev_b, stream);
            draw(out);
            std::swap(dev_a, dev_b);
        }
        else
        {
            break;
        }
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaStreamDestroy(stream); //"opaque type"
}

int main()
{
    std::vector<uint8_t> a;

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(false, true);

    for (int i = 0; i < width * height; ++i) {
        a.push_back(distrib(gen));
    }

    //a.resize(width * height);
    //a[0] = 1;
    //a[width] = 1;
    //a[width + 2] = 1;
    //a[width * 2] = 1;
    //a[width * 2 + 1] = 1;

    std::vector<uint8_t> b(a.size(), 0);
    app(a, b);
    cudaLogFatalError(cudaDeviceReset());
    return 0;
}

