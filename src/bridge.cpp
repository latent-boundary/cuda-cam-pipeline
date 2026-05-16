#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_kernels.h"

int main()
{
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    if (!cap.isOpened()) {
        printf("Camera not found\n");
        return -1;
    }

    cv::Mat frame, gray;
    cap >> frame;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    int width = gray.cols;
    int height = gray.rows;
    size_t size = width * height * sizeof(unsigned char);

    unsigned char *d_src, *d_dst, *d_tmp;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_tmp, size);
    cudaMalloc(&d_dst, size);

    int frame_id = 0;

    // FPS measurement
    cv::TickMeter tm;

    // Buffers for frame difference (motion factor)
    unsigned char *d_prev, *d_diff;
    cudaMalloc(&d_prev, size);
    cudaMalloc(&d_diff, size);

    // Initialize d_prev only for the first frame
    cudaMemcpy(d_prev, gray.data, size, cudaMemcpyHostToDevice);

    while (true) {
        // Start FPS timer
        tm.start();

        // Capture frame
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Copy grayscale frame to GPU
        cudaError_t err = cudaMemcpy(d_src, gray.data, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Memcpy H→D failed: %s\n", cudaGetErrorString(err));
        }

        // 1: Gaussian blur (d_src → d_tmp)
        launch_gaussian5x5(d_src, d_tmp, width, height);

        // 2: Sobel edge detection (d_tmp → d_dst)
        launch_sobel(d_tmp, d_dst, width, height);

        // 3: Frame difference (motion factor)
        launch_frame_diff(d_prev, d_tmp, d_diff, width, height);
        cudaDeviceSynchronize();

        // Update previous frame buffer
        cudaMemcpy(d_prev, d_tmp, size, cudaMemcpyDeviceToDevice);

        // Copy edge result back to CPU
        cv::Mat edge(height, width, CV_8UC1);
        cudaMemcpy(edge.data, d_dst, size, cudaMemcpyDeviceToHost);

        // Stop FPS timer and print
        tm.stop();
        printf("FPS: %.2f\n", 1.0 / tm.getTimeSec());
        tm.reset();

        // Save edge image every 100 frames
        if (frame_id % 100 == 0) {
            cv::imwrite("edge_" + std::to_string(frame_id) + ".jpg", edge);
            printf("Saved edge_%d.jpg\n", frame_id);
        }

        // Copy motion map to CPU and save
        cv::Mat diff(height, width, CV_8UC1);
        cudaMemcpy(diff.data, d_diff, size, cudaMemcpyDeviceToHost);

        if (frame_id % 100 == 0) {
            cv::imwrite("diff_" + std::to_string(frame_id) + ".jpg", diff);
        }

        frame_id++;
    }

    cudaFree(d_src);
    cudaFree(d_tmp);
    cudaFree(d_dst);
    cudaFree(d_prev);
    cudaFree(d_diff);

    return 0;
}
