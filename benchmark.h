#ifndef BENCHMARK_H
#define BENCHMARK_H

#define GLEW_STATIC
#include <GL/glew.h>

#include <vector>
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
#include "types.h"

// Run resolution benchmark function declaration
void runResolutionBenchmark(
    cv::VideoCapture &cap,
    GLFWwindow *window,
    unsigned int *shaders,
    unsigned int VAO,
    unsigned int texture,
    std::vector<PerformanceData> &performanceLog);

// Run automated benchmark function declaration with optional transform testing
void runAutomatedBenchmark(
    cv::VideoCapture &cap,
    GLFWwindow *window,
    unsigned int *shaders,
    unsigned int VAO,
    unsigned int texture,
    std::vector<PerformanceData> &performanceLog,
    bool testWithTransforms = false);

#endif