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

    //FPS測定
    cv::TickMeter tm;

    while (true) {
        //FPS測定開始
        tm.start();

        // パイプライン処理
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // h_gray → d_src にコピー
        cudaError_t err = cudaMemcpy(d_src, gray.data, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Memcpy H→D failed: %s\n", cudaGetErrorString(err));
        }

        // 1 : Gaussian: d_src → d_tmp
        launch_gaussian5x5(d_src, d_tmp, width, height);

        // 2: Sobel: d_tmp → d_dst
        launch_sobel(d_tmp, d_dst, width, height);

        //  Gaussian → Sobel の順序保証
        //  Check Kernel Errカーネルエラーが即座に検出
        cudaDeviceSynchronize();

        // d_dst → h_edge にコピー
        cv::Mat edge(height, width, CV_8UC1);
        cudaMemcpy(edge.data, d_dst, size, cudaMemcpyDeviceToHost);

        //FPS表示
        //FPS測定タイマ停止　→　FPS表示 →　FPS再開
        tm.stop();
        printf("FPS: %.2f\n", 1.0 / tm.getTimeSec());
        tm.reset();

        if (frame_id % 100 == 0) {
            cv::imwrite("edge_" + std::to_string(frame_id) + ".jpg", edge);
            printf("Saved edge_%d.jpg\n", frame_id);
        }

        frame_id++;
    }


    cudaFree(d_src);
    cudaFree(d_tmp);
    cudaFree(d_dst);

    return 0;
}
