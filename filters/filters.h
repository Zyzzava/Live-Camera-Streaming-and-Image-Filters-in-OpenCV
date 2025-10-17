#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

namespace ImageFilters
{
    // Basic filters
    cv::Mat applyGrayscale(const cv::Mat &input);
    cv::Mat applyGaussianBlur(const cv::Mat &input, int kernelSize = 15);
    cv::Mat applyEdgeDetection(const cv::Mat &input);
    cv::Mat applyPixelation(const cv::Mat &input, int pixelSize = 10);
    cv::Mat applyComicArt(const cv::Mat &input);
}

#endif