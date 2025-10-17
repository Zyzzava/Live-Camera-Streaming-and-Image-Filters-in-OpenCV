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

    cv::Mat applyPixelation(const cv::Mat &input, int pixelSize)
    {
        cv::Mat temp, output;
        // Shrink
        cv::resize(input, temp, cv::Size(input.cols / pixelSize, input.rows / pixelSize), 0, 0, cv::INTER_LINEAR);
        // Enlarge
        cv::resize(temp, output, cv::Size(input.cols, input.rows), 0, 0, cv::INTER_NEAREST);
        return output;
    }

    cv::Mat applyComicArt(const cv::Mat &input)
    {
        cv::Mat gray, blurred, edges, colorReduced, output;

        // Edge detection
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        cv::medianBlur(gray, blurred, 7);
        cv::adaptiveThreshold(blurred, edges, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 2);
        cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);

        // Color reduction
        input.copyTo(colorReduced);
        int div = 64;
        for (int y = 0; y < colorReduced.rows; y++)
        {
            for (int x = 0; x < colorReduced.cols; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    colorReduced.at<cv::Vec3b>(y, x)[c] =
                        colorReduced.at<cv::Vec3b>(y, x)[c] / div * div + div / 2;
                }
            }
        }

        // Combine edges and color reduced image
        cv::bitwise_and(colorReduced, edges, output);

        return output;
    }
}