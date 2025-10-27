#ifndef UTILS_H
#define UTILS_H

#define GLEW_STATIC
#include <GL/glew.h>

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "types.h"

// Shader compilation
unsigned int compileShaderProgram(const char *vertexSource, const char *fragmentSource);

// CSV writing
void writePerformanceCSV(const std::vector<PerformanceData> &data, const std::string &filename);

// CPU transformation
cv::Mat applyCPUTransform(const cv::Mat &input, const TransformParams &params);

#endif