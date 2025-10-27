#ifndef TYPES_H
#define TYPES_H

#include <string>

// Transformation parameters
struct TransformParams
{
    float translateX = 0.0f;
    float translateY = 0.0f;
    float scale = 1.0f;
    float rotation = 0.0f;
};

// Mouse callback data
struct MouseData
{
    double lastX = 0.0;
    double lastY = 0.0;
    bool isDragging = false;
    bool isRotating = false;
    TransformParams transform;
};

// Performance measurement structure
struct PerformanceData
{
    std::string filterName;
    std::string executionMode;
    int resolution;
    bool hasTransform;
    double fps;
    double frameTime;
};

#endif