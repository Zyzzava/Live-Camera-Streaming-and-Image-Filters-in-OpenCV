#include "utils.h"
#include <fstream>
#include <iostream>

// Compile shader program from vertex and fragment shader sources
unsigned int compileShaderProgram(const char *vertexSource, const char *fragmentSource)
{
    // Compile vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Compile fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Link program
    unsigned int program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    // Clean up shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

// Check for shader compilation errors
void writePerformanceCSV(const std::vector<PerformanceData> &data, const std::string &filename)
{
    std::ofstream file(filename);
    file << "FilterName,ExecutionMode,Resolution,HasTransform,FPS,FrameTime_ms\n";

    for (const auto &entry : data)
    {
        file << entry.filterName << ","
             << entry.executionMode << ","
             << entry.resolution << ","
             << (entry.hasTransform ? "Yes" : "No") << ","
             << entry.fps << ","
             << entry.frameTime << "\n";
    }

    file.close();
    std::cout << "Performance data written to " << filename << std::endl;
}

// Apply CPU-based transformation using OpenCV
cv::Mat applyCPUTransform(const cv::Mat &input, const TransformParams &params)
{
    cv::Mat output = cv::Mat::zeros(input.size(), input.type());

    // Translation in pixels (convert from normalized -1 to 1 to pixel coords)
    float txPixels = params.translateX * input.cols / 2.0f;
    float tyPixels = -params.translateY * input.rows / 2.0f; // Flip Y back

    // Combined transformation: scale and rotate around center, then translate
    cv::Point2f center(input.cols / 2.0f, input.rows / 2.0f);

    // Create rotation matrix with scale (angle in degrees, scale)
    cv::Mat transform = cv::getRotationMatrix2D(center, params.rotation, params.scale);

    // Add translation
    transform.at<double>(0, 2) += txPixels;
    transform.at<double>(1, 2) += tyPixels;

    cv::warpAffine(input, output, transform, input.size(), cv::INTER_LINEAR);

    return output;
}