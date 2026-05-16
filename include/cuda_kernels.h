#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_sobel(unsigned char* d_src, unsigned char* d_dst,
                  int width, int height);

void launch_gaussian5x5(unsigned char* d_src, unsigned char* d_dst,
                        int width, int height);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H
