#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

#define GLEW_STATIC

#include <GL/glew.h>
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
#include "filters.h"
#include "shaders.h"
#include "utils.h"
#include "types.h"
#include "benchmark.h"
#include "input_handler.h"

using namespace std;
using namespace cv;

int main()
{
    // Initialize GLFW
    if (!glfwInit())
    {
        cerr << "Failed to initialize GLFW" << endl;
        return -1;
    }

    // glfw setup
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // needed for mac

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
    InputHandler::setupCallbacks(window, &mouseData);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        cerr << "Failed to initialize GLEW" << endl;
        glfwTerminate();
        return -1;
    }

    // Setting up shaders
    // Compile vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &Shaders::vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Compile fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &Shaders::fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Link shaders into a shader program
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Delete shaders as they're now linked into our program and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Create shader programs for all filters
    unsigned int normalShader = shaderProgram;
    unsigned int grayscaleShader = compileShaderProgram(Shaders::vertexShaderSource, Shaders::grayscaleShaderSource);
    unsigned int blurShader = compileShaderProgram(Shaders::vertexShaderSource, Shaders::blurShaderSource);
    unsigned int edgeShader = compileShaderProgram(Shaders::vertexShaderSource, Shaders::edgeShaderSource);
    unsigned int pixelationShader = compileShaderProgram(Shaders::vertexShaderSource, Shaders::pixelationShaderSource);
    unsigned int comicShader = compileShaderProgram(Shaders::vertexShaderSource, Shaders::comicShaderSource);

    // Initialize OpenGL
    int fbWidth, fbHeight;
    // Get framebuffer size
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    // Set viewport
    glViewport(0, 0, fbWidth, fbHeight);

    // Set clear color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Create a texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Vertex data for a quad
    float vertices[] = {
        // Positions and Texture Coords
        -1.0f, 1.0f, 0.0f, 0.0f, // Top-left
        1.0f, 1.0f, 1.0f, 0.0f,  // Top-right
        1.0f, -1.0f, 1.0f, 1.0f, // Bottom-right
        -1.0f, -1.0f, 0.0f, 1.0f // Bottom-left
    };

    unsigned int indices[] = {
        0, 1, 2, // First triangle
        2, 3, 0  // Second triangle
    };

    // Setup buffers and arrays for rendering
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Setup VBO and EBO - which store vertex data and indices
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
            cerr << "Error: Something wrong, not getting frames" << endl;
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