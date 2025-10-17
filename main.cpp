#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

#define GLEW_STATIC

#include <GL/glew.h>
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
#include "filters.h"

// Shader source code
// We'll add some modifications in addition to first iteration
// Supporting transformations
const char *vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;

    uniform vec2 uTranslate;
    uniform float uScale;
    uniform float uRotation; // Rotation in radians

    void main() {
        // Create rotation matrix
        float cosTheta = cos(uRotation);
        float sinTheta = sin(uRotation);
        mat2 rotationMatrix = mat2(
            cosTheta, -sinTheta,
            sinTheta, cosTheta
        );
        
        // Apply transformations: scale -> rotate -> translate
        vec2 pos = aPos * uScale;
        pos = rotationMatrix * pos;
        pos = pos + uTranslate;
        
        gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
)";

const char *fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D ourTexture;
    void main() {
        FragColor = texture(ourTexture, TexCoord);
    }
)";

const char *grayscaleShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D ourTexture;
    void main() {
        vec4 color = texture(ourTexture, TexCoord);
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
        FragColor = vec4(vec3(gray), 1.0);
    }
)";

const char *blurShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D ourTexture;
    uniform vec2 texelSize;
    
    void main() {
        vec4 result = vec4(0.0);
        
        // 5x5 Gaussian kernel weights
        float kernel[25] = float[](
            1.0/273.0, 4.0/273.0, 7.0/273.0, 4.0/273.0, 1.0/273.0,
            4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0,
            7.0/273.0, 26.0/273.0, 41.0/273.0, 26.0/273.0, 7.0/273.0,
            4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0,
            1.0/273.0, 4.0/273.0, 7.0/273.0, 4.0/273.0, 1.0/273.0
        );
        
        int index = 0;
        for(int y = -2; y <= 2; y++) {
            for(int x = -2; x <= 2; x++) {
                vec2 offset = vec2(x, y) * texelSize;
                result += texture(ourTexture, TexCoord + offset) * kernel[index++];
            }
        }
        FragColor = result;
    }
)";

const char *edgeShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D ourTexture;
    uniform vec2 texelSize;
    
    void main() {
        // Sobel kernels for edge detection
        float Gx[9] = float[](-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
        float Gy[9] = float[](-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
        
        float edgeX = 0.0;
        float edgeY = 0.0;
        int index = 0;
        
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                vec2 offset = vec2(x, y) * texelSize;
                vec4 color = texture(ourTexture, TexCoord + offset);
                float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                edgeX += gray * Gx[index];
                edgeY += gray * Gy[index];
                index++;
            }
        }
        
        float edge = sqrt(edgeX * edgeX + edgeY * edgeY);
        // Threshold to make edges more visible (similar to Canny)
        edge = edge > 0.3 ? 1.0 : 0.0;
        FragColor = vec4(vec3(edge), 1.0);
    }
)";

const char *pixelationShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D ourTexture;
    uniform float pixelSize;
    uniform vec2 resolution;
    
    void main() {
        // Calculate pixel block size
        vec2 blockSize = vec2(pixelSize) / resolution;
        
        // Snap to pixel grid
        vec2 coord = floor(TexCoord / blockSize) * blockSize;
        
        FragColor = texture(ourTexture, coord);
    }
)";

const char *comicShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D ourTexture;
    uniform vec2 texelSize;
    
    void main() {
        // Edge detection using Sobel
        float Gx[9] = float[](-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
        float Gy[9] = float[](-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
        
        float edgeX = 0.0;
        float edgeY = 0.0;
        int index = 0;
        
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                vec2 offset = vec2(x, y) * texelSize;
                vec4 color = texture(ourTexture, TexCoord + offset);
                float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
                edgeX += gray * Gx[index];
                edgeY += gray * Gy[index];
                index++;
            }
        }
        
        float edge = sqrt(edgeX * edgeX + edgeY * edgeY);
        float edgeMask = edge > 0.2 ? 0.0 : 1.0;
        
        // Color quantization (reduce colors like comic art)
        vec4 color = texture(ourTexture, TexCoord);
        float levels = 4.0; // Number of color levels
        vec3 quantized = floor(color.rgb * levels) / levels + 0.5 / levels;
        
        // Combine edges with quantized color
        FragColor = vec4(quantized * edgeMask, 1.0);
    }
)";

using namespace std;
using namespace cv;

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
    string filterName;
    string executionMode;
    int resolution;
    bool hasTransform;
    double fps;
    double frameTime; // ms
};

// Writing to CSV
void writePerformanceCSV(const vector<PerformanceData> &data, const string &filename)
{
    ofstream file(filename);
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
    cout << "Performance data written to " << filename << endl;
}

// Mouse button callback
void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    MouseData *data = static_cast<MouseData *>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if (action == GLFW_PRESS)
        {
            data->isDragging = true;
            glfwGetCursorPos(window, &data->lastX, &data->lastY);
        }
        else if (action == GLFW_RELEASE)
        {
            data->isDragging = false;
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        if (action == GLFW_PRESS)
        {
            data->isRotating = true;
            glfwGetCursorPos(window, &data->lastX, &data->lastY);
        }
        else if (action == GLFW_RELEASE)
        {
            data->isRotating = false;
        }
    }
}

// Mouse movement callback
void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    MouseData *data = static_cast<MouseData *>(glfwGetWindowUserPointer(window));

    if (data->isDragging)
    {
        double dx = xpos - data->lastX;
        double dy = ypos - data->lastY;

        int width, height;
        glfwGetWindowSize(window, &width, &height);

        // Convert pixel movement to normalized coordinates (-1 to 1)
        data->transform.translateX += (float)(dx / width) * 2.0f;
        data->transform.translateY -= (float)(dy / height) * 2.0f; // Flip Y

        data->lastX = xpos;
        data->lastY = ypos;
    }
    else if (data->isRotating)
    {
        double dx = xpos - data->lastX;

        // Horizontal mouse movement controls rotation
        // Sensitivity: 0.5 degrees per pixel
        data->transform.rotation += (float)dx * 0.5f;

        data->lastX = xpos;
        data->lastY = ypos;
    }
}

// Mouse scroll callback for zoom
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    MouseData *data = static_cast<MouseData *>(glfwGetWindowUserPointer(window));
    data->transform.scale += (float)yoffset * 0.1f;
    // Clamp scale between 0.1 and 5.0
    data->transform.scale = max(0.1f, min(data->transform.scale, 5.0f));
}

// CPU-based transformation using openCV
Mat applyCPUTransform(const Mat &input, const TransformParams &params)
{
    Mat output = Mat::zeros(input.size(), input.type());

    // Translation in pixels (convert from normalized -1 to 1 to pixel coords)
    float txPixels = params.translateX * input.cols / 2.0f;
    float tyPixels = -params.translateY * input.rows / 2.0f; // Flip Y back

    // Combined transformation: scale and rotate around center, then translate
    Point2f center(input.cols / 2.0f, input.rows / 2.0f);

    // Create rotation matrix with scale (angle in degrees, scale)
    Mat transform = getRotationMatrix2D(center, params.rotation, params.scale);

    // Add translation
    transform.at<double>(0, 2) += txPixels;
    transform.at<double>(1, 2) += tyPixels;

    warpAffine(input, output, transform, input.size(), INTER_LINEAR);

    return output;
}

void runResolutionBenchmark(
    VideoCapture &cap,
    GLFWwindow *window,
    unsigned int *shaders,
    unsigned int VAO,
    unsigned int texture,
    vector<PerformanceData> &performanceLog)
{
    const int BENCHMARK_FRAMES = 50;
    const char *filterNames[] = {"None", "Grayscale", "Blur", "Edge", "Pixelation", "Comic"};
    const int numFilters = 6;
    const bool executionModes[] = {true, false}; // GPU first, then CPU
    const char *modeNames[] = {"GPU", "CPU"};

    // Define resolutions to test
    struct Resolution
    {
        int width;
        int height;
        string name;
    };

    vector<Resolution> resolutions = {
        {640, 480, "VGA (640x480)"},
        {1280, 720, "HD (1280x720)"},
        {1920, 1080, "Full HD (1920x1080)"}};

    // Store original resolution
    int originalWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int originalHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

    int totalTests = numFilters * 2 * resolutions.size();
    int currentTest = 0;

    cout << "\n========================================" << endl;
    cout << "STARTING RESOLUTION BENCHMARK" << endl;
    cout << "Total tests: " << totalTests << endl;
    cout << "Frames per test: " << BENCHMARK_FRAMES << endl;
    cout << "Resolutions to test: " << resolutions.size() << endl;
    cout << "========================================\n"
         << endl;

    // No transform for resolution testing
    TransformParams noTransform;
    noTransform.translateX = 0.0f;
    noTransform.translateY = 0.0f;
    noTransform.scale = 1.0f;
    noTransform.rotation = 0.0f;

    // Loop through each resolution
    for (const auto &res : resolutions)
    {
        cout << "\n*** Testing Resolution: " << res.name << " ***" << endl;

        // Set camera resolution
        cap.set(CAP_PROP_FRAME_WIDTH, res.width);
        cap.set(CAP_PROP_FRAME_HEIGHT, res.height);

        // Verify resolution was set
        int actualWidth = cap.get(CAP_PROP_FRAME_WIDTH);
        int actualHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
        cout << "  Actual resolution: " << actualWidth << "x" << actualHeight << endl;

        // Update window size
        glfwSetWindowSize(window, actualWidth, actualHeight);

        // Update viewport
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);

        // Loop through execution modes (GPU, then CPU)
        for (int modeIdx = 0; modeIdx < 2; modeIdx++)
        {
            bool useGPU = executionModes[modeIdx];

            // Loop through all filters
            for (int filterIdx = 0; filterIdx < numFilters; filterIdx++)
            {
                currentTest++;
                cout << "[" << currentTest << "/" << totalTests << "] "
                     << filterNames[filterIdx] << " | " << modeNames[modeIdx]
                     << " | " << res.name << endl;

                vector<double> frameTimes;

                // Benchmark loop - collect frame times
                for (int frame = 0; frame < BENCHMARK_FRAMES; frame++)
                {
                    auto frameStart = chrono::high_resolution_clock::now();

                    // Capture frame
                    Mat capturedFrame;
                    cap >> capturedFrame;
                    if (capturedFrame.empty())
                    {
                        cerr << "Error: Could not read frame during benchmark!" << endl;
                        return;
                    }

                    Mat displayFrame;
                    unsigned int shaderToUse = shaders[0]; // Default shader

                    if (useGPU)
                    {
                        // GPU mode - raw frame, filter applied in shader
                        displayFrame = capturedFrame;
                        shaderToUse = shaders[filterIdx];
                    }
                    else
                    {
                        // CPU mode - apply filter with OpenCV
                        switch (filterIdx)
                        {
                        case 0:
                            displayFrame = capturedFrame;
                            break;
                        case 1:
                            displayFrame = ImageFilters::applyGrayscale(capturedFrame);
                            break;
                        case 2:
                            displayFrame = ImageFilters::applyGaussianBlur(capturedFrame, 5);
                            break;
                        case 3:
                            displayFrame = ImageFilters::applyEdgeDetection(capturedFrame);
                            break;
                        case 4:
                            displayFrame = ImageFilters::applyPixelation(capturedFrame, 10);
                            break;
                        case 5:
                            displayFrame = ImageFilters::applyComicArt(capturedFrame);
                            break;
                        }
                        // Apply CPU transform (no transform for resolution test)
                        displayFrame = applyCPUTransform(displayFrame, noTransform);
                    }

                    // Convert to RGB
                    Mat frame_rgb;
                    cvtColor(displayFrame, frame_rgb, COLOR_BGR2RGB);

                    // Update texture
                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.cols, frame_rgb.rows,
                                 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.data);

                    // Render
                    glClear(GL_COLOR_BUFFER_BIT);
                    glUseProgram(shaderToUse);

                    if (useGPU)
                    {
                        // Apply GPU transform (no transform)
                        GLint translateLoc = glGetUniformLocation(shaderToUse, "uTranslate");
                        GLint scaleLoc = glGetUniformLocation(shaderToUse, "uScale");
                        GLint rotationLoc = glGetUniformLocation(shaderToUse, "uRotation");

                        glUniform2f(translateLoc, noTransform.translateX, noTransform.translateY);
                        glUniform1f(scaleLoc, noTransform.scale);
                        glUniform1f(rotationLoc, noTransform.rotation * 3.14159265f / 180.0f);
                    }

                    // Set filter-specific uniforms
                    if (filterIdx == 2 || filterIdx == 3 || filterIdx == 5)
                    {
                        GLint texelLoc = glGetUniformLocation(shaderToUse, "texelSize");
                        glUniform2f(texelLoc, 1.0f / frame_rgb.cols, 1.0f / frame_rgb.rows);
                    }

                    if (filterIdx == 4)
                    {
                        GLint pixelLoc = glGetUniformLocation(shaderToUse, "pixelSize");
                        GLint resLoc = glGetUniformLocation(shaderToUse, "resolution");
                        glUniform1f(pixelLoc, 10.0f);
                        glUniform2f(resLoc, (float)frame_rgb.cols, (float)frame_rgb.rows);
                    }

                    glBindVertexArray(VAO);
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                    glfwSwapBuffers(window);
                    glfwPollEvents();

                    // Calculate frame time
                    auto frameEnd = chrono::high_resolution_clock::now();
                    double frameTime = chrono::duration<double, milli>(frameEnd - frameStart).count();
                    frameTimes.push_back(frameTime);

                    // Progress indicator every 25 frames
                    if ((frame + 1) % 25 == 0)
                    {
                        cout << "    Progress: " << (frame + 1) << "/" << BENCHMARK_FRAMES << " frames" << endl;
                    }
                }

                // Calculate statistics
                double avgFrameTime = 0.0;
                for (double ft : frameTimes)
                {
                    avgFrameTime += ft;
                }
                avgFrameTime /= frameTimes.size();
                double avgFPS = 1000.0 / avgFrameTime;

                // Log results
                PerformanceData data;
                data.filterName = filterNames[filterIdx];
                data.executionMode = modeNames[modeIdx];
                data.resolution = actualWidth * actualHeight;
                data.hasTransform = false; // No transforms for resolution testing
                data.fps = avgFPS;
                data.frameTime = avgFrameTime;
                performanceLog.push_back(data);

                cout << "    ✓ Avg FPS: " << fixed << setprecision(2) << avgFPS
                     << " | Frame Time: " << avgFrameTime << "ms\n"
                     << endl;
            }
        }
    }

    // Restore original resolution
    cap.set(CAP_PROP_FRAME_WIDTH, originalWidth);
    cap.set(CAP_PROP_FRAME_HEIGHT, originalHeight);
    glfwSetWindowSize(window, originalWidth, originalHeight);

    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    cout << "RESOLUTION BENCHMARK COMPLETE" << endl;
    cout << "Total measurements: " << performanceLog.size() << endl;
    cout << "Resolution restored to: " << originalWidth << "x" << originalHeight << endl;

    // Automatically save results
    writePerformanceCSV(performanceLog, "performance_resolution.csv");
    cout << "Results saved to performance_resolution.csv\n"
         << endl;
}

void runAutomatedBenchmark(
    VideoCapture &cap,
    GLFWwindow *window,
    unsigned int *shaders,
    unsigned int VAO,
    unsigned int texture,
    vector<PerformanceData> &performanceLog,
    bool testWithTransforms = false) // flag for transforms
{
    const int BENCHMARK_FRAMES = 50;
    const char *filterNames[] = {"None", "Grayscale", "Blur", "Edge", "Pixelation", "Comic"};
    const int numFilters = 6;
    const bool executionModes[] = {true, false}; // GPU first, then CPU
    const char *modeNames[] = {"GPU", "CPU"};

    // Define transform cfgs
    vector<TransformParams> transformConfigs;
    vector<string> transformNames;

    // No transform (baseline)
    TransformParams noTransform;
    noTransform.translateX = 0.0f;
    noTransform.translateY = 0.0f;
    noTransform.scale = 1.0f;
    noTransform.rotation = 0.0f;
    transformConfigs.push_back(noTransform);
    transformNames.push_back("No Transform");

    // Add various transform cfgs if requested
    if (testWithTransforms)
    {
        // Translation only
        TransformParams translateOnly;
        translateOnly.translateX = 0.3f;
        translateOnly.translateY = 0.2f;
        translateOnly.scale = 1.0f;
        translateOnly.rotation = 0.0f;
        transformConfigs.push_back(translateOnly);
        transformNames.push_back("Translation");

        // Scale only
        TransformParams scaleOnly;
        scaleOnly.translateX = 0.0f;
        scaleOnly.translateY = 0.0f;
        scaleOnly.scale = 1.5f;
        scaleOnly.rotation = 0.0f;
        transformConfigs.push_back(scaleOnly);
        transformNames.push_back("Scale");

        // Rotation only
        TransformParams rotateOnly;
        rotateOnly.translateX = 0.0f;
        rotateOnly.translateY = 0.0f;
        rotateOnly.scale = 1.0f;
        rotateOnly.rotation = 25.0f;
        transformConfigs.push_back(rotateOnly);
        transformNames.push_back("Rotation");

        // Combined transform
        TransformParams combined;
        combined.translateX = 0.2f;
        combined.translateY = -0.15f;
        combined.scale = 1.3f;
        combined.rotation = 15.0f;
        transformConfigs.push_back(combined);
        transformNames.push_back("Combined");
    }

    int totalTests = numFilters * 2 * transformConfigs.size();
    int currentTest = 0;

    cout << "STARTING AUTOMATED BENCHMARK" << endl;
    cout << "Total tests: " << totalTests << endl;
    cout << "Frames per test: " << BENCHMARK_FRAMES << endl;
    cout << "Transform configurations: " << transformConfigs.size() << endl;

    // Loop through each transform configuration
    for (size_t transformIdx = 0; transformIdx < transformConfigs.size(); transformIdx++)
    {
        const TransformParams &currentTransform = transformConfigs[transformIdx];
        const string &transformName = transformNames[transformIdx];

        cout << "\n*** Testing Transform Config: " << transformName << " ***" << endl;
        cout << "  Translation: (" << currentTransform.translateX << ", " << currentTransform.translateY << ")" << endl;
        cout << "  Scale: " << currentTransform.scale << endl;
        cout << "  Rotation: " << currentTransform.rotation << "°\n"
             << endl;

        // Loop through execution modes (GPU, then CPU)
        for (int modeIdx = 0; modeIdx < 2; modeIdx++)
        {
            bool useGPU = executionModes[modeIdx];

            // Loop through all filters
            for (int filterIdx = 0; filterIdx < numFilters; filterIdx++)
            {
                currentTest++;
                cout << "[" << currentTest << "/" << totalTests << "] "
                     << filterNames[filterIdx] << " | " << modeNames[modeIdx]
                     << " | " << transformName << endl;

                vector<double> frameTimes;

                // Benchmark loop - collect frame times
                for (int frame = 0; frame < BENCHMARK_FRAMES; frame++)
                {
                    auto frameStart = chrono::high_resolution_clock::now();

                    // Capture frame
                    Mat capturedFrame;
                    cap >> capturedFrame;
                    if (capturedFrame.empty())
                    {
                        cerr << "Error: Could not read frame during benchmark!" << endl;
                        return;
                    }

                    Mat displayFrame;
                    unsigned int shaderToUse = shaders[0]; // Default shader

                    if (useGPU)
                    {
                        // GPU mode - raw frame, filter applied in shader
                        displayFrame = capturedFrame;
                        shaderToUse = shaders[filterIdx];
                    }
                    else
                    {
                        // CPU mode - apply filter with OpenCV
                        switch (filterIdx)
                        {
                        case 0:
                            displayFrame = capturedFrame;
                            break;
                        case 1:
                            displayFrame = ImageFilters::applyGrayscale(capturedFrame);
                            break;
                        case 2:
                            displayFrame = ImageFilters::applyGaussianBlur(capturedFrame, 5);
                            break;
                        case 3:
                            displayFrame = ImageFilters::applyEdgeDetection(capturedFrame);
                            break;
                        case 4:
                            displayFrame = ImageFilters::applyPixelation(capturedFrame, 10);
                            break;
                        case 5:
                            displayFrame = ImageFilters::applyComicArt(capturedFrame);
                            break;
                        }
                        // Apply CPU transform (same transform for fair comparison)
                        displayFrame = applyCPUTransform(displayFrame, currentTransform);
                    }

                    // Convert to RGB
                    Mat frame_rgb;
                    cvtColor(displayFrame, frame_rgb, COLOR_BGR2RGB);

                    // Update texture
                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.cols, frame_rgb.rows,
                                 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.data);

                    // Render
                    glClear(GL_COLOR_BUFFER_BIT);
                    glUseProgram(shaderToUse);

                    if (useGPU)
                    {
                        // Apply GPU transform (same transform as CPU for fair comparison)
                        GLint translateLoc = glGetUniformLocation(shaderToUse, "uTranslate");
                        GLint scaleLoc = glGetUniformLocation(shaderToUse, "uScale");
                        GLint rotationLoc = glGetUniformLocation(shaderToUse, "uRotation");

                        glUniform2f(translateLoc, currentTransform.translateX, currentTransform.translateY);
                        glUniform1f(scaleLoc, currentTransform.scale);
                        glUniform1f(rotationLoc, currentTransform.rotation * 3.14159265f / 180.0f);
                    }

                    // Set filter-specific uniforms
                    if (filterIdx == 2 || filterIdx == 3 || filterIdx == 5)
                    {
                        GLint texelLoc = glGetUniformLocation(shaderToUse, "texelSize");
                        glUniform2f(texelLoc, 1.0f / frame_rgb.cols, 1.0f / frame_rgb.rows);
                    }

                    if (filterIdx == 4)
                    {
                        GLint pixelLoc = glGetUniformLocation(shaderToUse, "pixelSize");
                        GLint resLoc = glGetUniformLocation(shaderToUse, "resolution");
                        glUniform1f(pixelLoc, 10.0f);
                        glUniform2f(resLoc, (float)frame_rgb.cols, (float)frame_rgb.rows);
                    }

                    glBindVertexArray(VAO);
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                    glfwSwapBuffers(window);
                    glfwPollEvents();

                    // Calculate frame time
                    auto frameEnd = chrono::high_resolution_clock::now();
                    double frameTime = chrono::duration<double, milli>(frameEnd - frameStart).count();
                    frameTimes.push_back(frameTime);

                    // Progress indicator every 100 frames
                    if ((frame + 1) % 100 == 0)
                    {
                        cout << "    Progress: " << (frame + 1) << "/" << BENCHMARK_FRAMES << " frames" << endl;
                    }
                }

                // Calculate statistics
                double avgFrameTime = 0.0;
                for (double ft : frameTimes)
                {
                    avgFrameTime += ft;
                }
                avgFrameTime /= frameTimes.size();
                double avgFPS = 1000.0 / avgFrameTime;

                // Determine if transform is active
                bool hasTransform = (transformIdx > 0);

                // Log results
                PerformanceData data;
                data.filterName = filterNames[filterIdx];
                data.executionMode = modeNames[modeIdx];
                data.resolution = cap.get(CAP_PROP_FRAME_WIDTH) * cap.get(CAP_PROP_FRAME_HEIGHT);
                data.hasTransform = hasTransform;
                data.fps = avgFPS;
                data.frameTime = avgFrameTime;
                performanceLog.push_back(data);

                cout << "avg FPS: " << fixed << setprecision(2) << avgFPS
                     << " | Frame Time: " << avgFrameTime << "ms\n"
                     << endl;
            }
        }
    }

    cout << "AUTOMATED BENCHMARK COMPLETE" << endl;
    cout << "Total measurements: " << performanceLog.size() << endl;

    // Automatically save results
    writePerformanceCSV(performanceLog, "performance_transforms.csv");
    cout << "saved to performance_transforms.csv" << endl;
}

int main()
{
    // Initialize GLFW
    if (!glfwInit())
    {
        cerr << "Failed to initialize GLFW" << endl;
        return -1;
    }

    // GLFW window hints to use OpenGL 3.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required for macOS

    // Create VideoCapture object
    VideoCapture cap(1);
    if (!cap.isOpened())
    {
        cerr << "Error: Could not open camera!" << endl;
        glfwTerminate();
        return -1;
    }
    int camWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int camHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

    // Create a GLFW window
    GLFWwindow *window = glfwCreateWindow(camWidth, camHeight, "OpenGL Webcam", NULL, NULL);
    if (!window)
    {
        cerr << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);

    // Initialize mouse data and set callbacks
    MouseData mouseData;
    glfwSetWindowUserPointer(window, &mouseData);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        cerr << "Failed to initialize GLEW" << endl;
        glfwTerminate();
        return -1;
    }

    // --- SHADER SETUP ---
    // Compile vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // NOTE: You should add error checking for shader compilation in a real application

    // Compile fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // NOTE: You should add error checking for shader compilation in a real application

    // Link shaders into a shader program
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Delete shaders as they're now linked into our program and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Create shader programs for both normal and grayscale rendering
    unsigned int normalShader = shaderProgram;
    unsigned int grayscaleShader = compileShaderProgram(vertexShaderSource, grayscaleShaderSource);
    unsigned int blurShader = compileShaderProgram(vertexShaderSource, blurShaderSource);
    unsigned int edgeShader = compileShaderProgram(vertexShaderSource, edgeShaderSource);
    unsigned int pixelationShader = compileShaderProgram(vertexShaderSource, pixelationShaderSource);
    unsigned int comicShader = compileShaderProgram(vertexShaderSource, comicShaderSource);

    // Initialize OpenGL
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Create a texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Vertex data for a quad
    float vertices[] = {
        // Positions   // Texture coordinates
        -1.0f, 1.0f, 0.0f, 0.0f, // Top-left
        1.0f, 1.0f, 1.0f, 0.0f,  // Top-right
        1.0f, -1.0f, 1.0f, 1.0f, // Bottom-right
        -1.0f, -1.0f, 0.0f, 1.0f // Bottom-left
    };

    unsigned int indices[] = {
        0, 1, 2, // First triangle
        2, 3, 0  // Second triangle
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    Mat frame;
    int currentFilter = 0;
    bool useGPU = true; // Toggle between GPU (true) and CPU (false) rendering

    // Performance tracking variables
    vector<PerformanceData> resolutionLog;
    vector<PerformanceData> transformLog;
    vector<PerformanceData> manualLog;

    auto lastTime = chrono::high_resolution_clock::now();
    int frameCount = 0;
    double fps = 0.0;
    vector<double> frameTimes;

    // Benchmark mode flag
    bool benchmarkMode = false; // enable benchmarking
    int benchmarkFrames = 0;
    const int BENCHMARK_DURATION = 300; // no. oF frames

    // array off shaders for the automated benchmark
    unsigned int shaders[] = {
        normalShader,
        grayscaleShader,
        blurShader,
        edgeShader,
        pixelationShader,
        comicShader};

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        auto frameStart = chrono::high_resolution_clock::now();

        // Capture frame
        cap >> frame;
        if (frame.empty())
        {
            cerr << "Error: Could not read frame!" << endl;
            break;
        }

        // Apply filter
        Mat displayFrame;
        unsigned int shaderToUse = normalShader; // Default to normal shader

        if (useGPU)
        {
            // GPU rendering - use shader, no CPU processing
            displayFrame = frame;

            // Select appropriate shader based on filter
            switch (currentFilter)
            {
            case 1:
                shaderToUse = grayscaleShader;
                break;
            case 2:
                shaderToUse = blurShader;
                break;
            case 3:
                shaderToUse = edgeShader;
                break;
            case 4:
                shaderToUse = pixelationShader;
                break;
            case 5:
                shaderToUse = comicShader;
                break;
            default:
                shaderToUse = normalShader;
                break;
            }
        }
        else
        {
            switch (currentFilter)
            {
            case 1:
                displayFrame = ImageFilters::applyGrayscale(frame);
                break;
            case 2:
                displayFrame = ImageFilters::applyGaussianBlur(frame, 5);
                break;
            case 3:
                displayFrame = ImageFilters::applyEdgeDetection(frame);
                break;
            case 4:
                displayFrame = ImageFilters::applyPixelation(frame, 10);
                break;
            case 5:
                displayFrame = ImageFilters::applyComicArt(frame);
                break;
            default:
                displayFrame = frame;
                break;
            }
            // Apply CPU-based transformations
            displayFrame = applyCPUTransform(displayFrame, mouseData.transform);
        }

        // Convert frame to RGB
        Mat frame_rgb;
        cvtColor(displayFrame, frame_rgb, COLOR_BGR2RGB);

        // Update texture
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.cols, frame_rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.data);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderToUse);

        // Set transformation uniforms (GPU mode)
        if (useGPU)
        {
            GLint translateLoc = glGetUniformLocation(shaderToUse, "uTranslate");
            GLint scaleLoc = glGetUniformLocation(shaderToUse, "uScale");
            GLint rotationLoc = glGetUniformLocation(shaderToUse, "uRotation");

            glUniform2f(translateLoc, mouseData.transform.translateX, mouseData.transform.translateY);
            glUniform1f(scaleLoc, mouseData.transform.scale);
            glUniform1f(rotationLoc, mouseData.transform.rotation * 3.14159265f / 180.0f); // Convert degrees to radians
        }

        // Set uniforms for shaders that need them
        if (currentFilter == 2 || currentFilter == 3 || currentFilter == 5)
        {
            // Blur, Edge, and Comic shaders need texel size
            GLint texelLoc = glGetUniformLocation(shaderToUse, "texelSize");
            glUniform2f(texelLoc, 1.0f / frame_rgb.cols, 1.0f / frame_rgb.rows);
        }

        if (currentFilter == 4)
        {
            // Pixelation shader needs pixel size and resolution
            GLint pixelLoc = glGetUniformLocation(shaderToUse, "pixelSize");
            GLint resLoc = glGetUniformLocation(shaderToUse, "resolution");
            glUniform1f(pixelLoc, 10.0f); // Pixel block size
            glUniform2f(resLoc, (float)frame_rgb.cols, (float)frame_rgb.rows);
        }

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Calculate the frame time
        auto frameEnd = chrono::high_resolution_clock::now();
        double frameTime = chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        frameTimes.push_back(frameTime);
        frameCount++;

        // Now we find FPS
        auto currentTime = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(currentTime - lastTime).count();

        if (elapsed >= 1.0)
        {
            fps = frameCount / elapsed;
            cout << "FPS: " << fps << " | Frame Time: " << frameTime << " ms" << endl;
            frameCount = 0;
            lastTime = currentTime;
        }

        // Benchmarking logic
        if (benchmarkMode)
        {
            benchmarkFrames++;
            if (benchmarkFrames >= BENCHMARK_DURATION)
            {
                // avg fps and frame time
                double avgFrameTime = 0.0;
                for (double ft : frameTimes)
                {
                    avgFrameTime += ft;
                }
                avgFrameTime /= frameTimes.size();
                double avgFPS = 1000.0 / avgFrameTime;

                //  the name of the filter
                string filterName;
                switch (currentFilter)
                {
                default:
                    filterName = "Normal";
                    break;
                case 1:
                    filterName = "Grayscale";
                    break;
                case 2:
                    filterName = "Gaussian Blur";
                    break;
                case 3:
                    filterName = "Edge Detection";
                    break;
                case 4:
                    filterName = "Pixelation";
                    break;
                case 5:
                    filterName = "Comic Art";
                    break;
                }
                // is transform active
                bool hasTransform = (mouseData.transform.translateX != 0.0f ||
                                     mouseData.transform.translateY != 0.0f ||
                                     mouseData.transform.scale != 1.0f ||
                                     mouseData.transform.rotation != 0.0f);

                // logging
                PerformanceData data;
                data.filterName = filterName;
                data.executionMode = useGPU ? "GPU" : "CPU";
                data.resolution = frame.cols * frame.rows;
                data.hasTransform = hasTransform;
                data.fps = avgFPS;
                data.frameTime = avgFrameTime;
                manualLog.push_back(data);

                cout << "Benchmark Complete" << endl;
                cout << "Filter: " << filterName << endl;
                cout << "Mode: " << data.executionMode << endl;
                cout << "Avg FPS: " << avgFPS << endl;
                cout << "Avg Frame Time: " << avgFrameTime << "ms" << endl;

                // Reset for next benchmark
                benchmarkMode = false;
                benchmarkFrames = 0;
                frameTimes.clear();
            }
        }

        // Handle key presses
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
            currentFilter = 1;
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
            currentFilter = 2;
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
            currentFilter = 3;
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
            currentFilter = 4;
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS)
            currentFilter = 5;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            currentFilter = 0;
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
            useGPU = !useGPU; // Toggle GPU/CPU rendering
        if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
        {
            benchmarkMode = true; // Start benchmarking
            benchmarkFrames = 0;
            frameTimes.clear();
            cout << "Starting manual benchmark..." << endl;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            writePerformanceCSV(manualLog, "performance_manual.csv"); // saved
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            cout << "Starting basic auto benchmark, no transforms..." << endl;
            transformLog.clear();
            runAutomatedBenchmark(cap, window, shaders, VAO, texture, transformLog, false);
            cout << "Automated benchmark completed." << endl;
        }
        if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
        {
            cout << "Starting advanced auto benchmark with transforms..." << endl;
            transformLog.clear();
            runAutomatedBenchmark(cap, window, shaders, VAO, texture, transformLog, true);
            cout << "Automated benchmark completed." << endl;
        }
        if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS)
        {
            cout << "Starting resolution benchmark..." << endl;
            resolutionLog.clear();
            runResolutionBenchmark(cap, window, shaders, VAO, texture, resolutionLog);
            cout << "Resolution benchmark completed." << endl;
        }
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        {
            // Reset transformation
            mouseData.transform.translateX = 0.0f;
            mouseData.transform.translateY = 0.0f;
            mouseData.transform.scale = 1.0f;
            mouseData.transform.rotation = 0.0f;
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            break;
    }

    // Cleanup
    glDeleteProgram(shaderProgram);
    glDeleteProgram(normalShader);
    glDeleteProgram(grayscaleShader);
    glDeleteProgram(blurShader);
    glDeleteProgram(edgeShader);
    glDeleteProgram(pixelationShader);
    glDeleteProgram(comicShader);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteTextures(1, &texture);
    glfwTerminate();
    cap.release();

    return 0;
}