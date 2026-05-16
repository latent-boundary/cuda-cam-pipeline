#include <cuda_runtime.h>

__global__
void sobel_shared_kernel(const unsigned char* src, unsigned char* dst,
                         int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int gx = 0, gy = 0;

    int SOBEL_GX[9] = {-1,0,1,-2,0,2,-1,0,1};
    int SOBEL_GY[9] = {-1,-2,-1,0,0,0,1,2,1};

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int ix = min(max(x + kx, 0), width - 1);
            int iy = min(max(y + ky, 0), height - 1);

            unsigned char v = src[iy * width + ix];
            int k = (ky + 1) * 3 + (kx + 1);

            gx += SOBEL_GX[k] * v;
            gy += SOBEL_GY[k] * v;
        }
    }

    int mag = abs(gx) + abs(gy);
    if (mag > 255) mag = 255;

    dst[y * width + x] = (unsigned char)mag;
}

// ★★★ これが bridge.cpp から呼ばれる関数 ★★★
extern "C" void launch_sobel(unsigned char* d_src, unsigned char* d_dst,
                             int width, int height)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    sobel_shared_kernel<<<grid, block>>>(d_src, d_dst, width, height);
    cudaDeviceSynchronize();
}
