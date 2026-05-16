#include <cuda_runtime.h>

__constant__ float GAUSS5[25] = {
    1,  4,  6,  4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1,  4,  6,  4, 1
};

__global__
void gaussian5x5_shared_kernel(const unsigned char* src, unsigned char* dst,
                               int width, int height)
{
    __shared__ unsigned char tile[16 + 4][16 + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * 16 + tx;
    int y = blockIdx.y * 16 + ty;

    int load_x = x - 2;
    int load_y = y - 2;

    load_x = max(0, min(load_x, width - 1));
    load_y = max(0, min(load_y, height - 1));

    tile[ty][tx] = src[load_y * width + load_x];

    __syncthreads();

    if (tx < 16 && ty < 16 && x < width && y < height) {
        float sum = 0.0f;

        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                sum += GAUSS5[ky * 5 + kx] * tile[ty + ky][tx + kx];
            }
        }

        sum /= 256.0f;

        dst[y * width + x] = (unsigned char)sum;
    }
}

extern "C" void launch_gaussian5x5(unsigned char* d_src, unsigned char* d_dst,
                                   int width, int height)
{
    dim3 block(16 + 4, 16 + 4);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    gaussian5x5_shared_kernel<<<grid, block>>>(d_src, d_dst, width, height);
    cudaDeviceSynchronize();
}
