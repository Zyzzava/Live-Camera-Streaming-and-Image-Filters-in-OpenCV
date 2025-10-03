#include "filters.h"
#include <opencv2/opencv.hpp>

namespace ImageFilters
{

    cv::Mat applyGrayscale(const cv::Mat &input)
    {
        cv::Mat output;
        cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR); // Convert back to 3 channels for consistency
        return output;
    }

    cv::Mat applyGaussianBlur(const cv::Mat &input, int kernelSize)
    {
        cv::Mat output;
        // Ensure kernel size is odd
        if (kernelSize % 2 == 0)
            kernelSize++;
        cv::GaussianBlur(input, output, cv::Size(kernelSize, kernelSize), 0);
        return output;
    }

    cv::Mat applyEdgeDetection(const cv::Mat &input)
    {
        cv::Mat gray, output;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, gray, 100, 200);
        cv::cvtColor(gray, output, cv::COLOR_GRAY2BGR);
        return output;
    }
}