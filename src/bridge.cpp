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
    size_t size = width * height;

    unsigned char *d_src, *d_dst;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);

    int frame_id = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cudaMemcpy(d_src, gray.data, size, cudaMemcpyHostToDevice);

        // CUDA カーネル呼び出し
        launch_sobel(d_src, d_dst, width, height);

        cv::Mat edge(height, width, CV_8UC1);
        cudaMemcpy(edge.data, d_dst, size, cudaMemcpyDeviceToHost);

        if (frame_id % 100 == 0) {
            cv::imwrite("edge_" + std::to_string(frame_id) + ".jpg", edge);
            printf("Saved edge_%d.jpg\n", frame_id);
        }

        frame_id++;
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
