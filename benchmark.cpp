#define GLEW_STATIC

#include "benchmark.h"
#include "filters.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <GL/glew.h>

using namespace std;
using namespace cv;

void runResolutionBenchmark(
    // Video capture object
    VideoCapture &cap,
    // OpenGL window
    GLFWwindow *window,
    // Shader programs
    unsigned int *shaders,
    // Vertex Array Object
    unsigned int VAO,
    // Texture
    unsigned int texture,
    // Performance log
    vector<PerformanceData> &performanceLog)
{
    // Benchmarking logic
    // Setting up benchmarking frames and specifying filter/mode names.
    const int BENCHMARK_FRAMES = 50;
    const char *filterNames[] = {"None", "Grayscale", "Blur", "Edge", "Pixelation", "Comic"};
    const int numFilters = 6;
    const bool executionModes[] = {true, false};
    const char *modeNames[] = {"GPU", "CPU"};

    // Struct for resolution configurations
    struct Resolution
    {
        int width;
        int height;
        string name;
    };

    // List of resolutions to test
    vector<Resolution> resolutions = {
        {640, 480, "VGA (640x480)"},
        {1280, 720, "HD (1280x720)"},
        {1920, 1080, "Full HD (1920x1080)"}};

    // Saving original resolution w and h
    int originalWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int originalHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

    // Total tests calculation
    int totalTests = numFilters * 2 * resolutions.size();

    int currentTest = 0;

    cout << "Resolution benchmark" << endl;
    cout << "Total tests: " << totalTests << endl;
    cout << "Frames per test: " << BENCHMARK_FRAMES << endl;
    cout << "Resolutions to test: " << resolutions.size() << endl;

    // No transform for resolution benchmark
    TransformParams noTransform;
    noTransform.translateX = 0.0f;
    noTransform.translateY = 0.0f;
    noTransform.scale = 1.0f;
    noTransform.rotation = 0.0f;

    // Going through each resolution
    for (const auto &res : resolutions)
    {
        cout << "Testing Resolution: " << res.name << endl;

        // Setting the capture resolution
        cap.set(CAP_PROP_FRAME_WIDTH, res.width);
        cap.set(CAP_PROP_FRAME_HEIGHT, res.height);

        // Getting the actual resolution set
        int actualWidth = cap.get(CAP_PROP_FRAME_WIDTH);
        int actualHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
        cout << "Actual resolution: " << actualWidth << "x" << actualHeight << endl;

        // Adjusting the window size
        glfwSetWindowSize(window, actualWidth, actualHeight);

        // Adjusting the viewport
        int fbWidth, fbHeight;
        // Setting the viewport to match new framebuffer size
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        // Set viewport to new size
        glViewport(0, 0, fbWidth, fbHeight);

        // Going through each execution mode and filter
        for (int modeIdx = 0; modeIdx < 2; modeIdx++)
        {
            bool useGPU = executionModes[modeIdx];

            // Going through each filter
            for (int filterIdx = 0; filterIdx < numFilters; filterIdx++)
            {
                currentTest++;
                cout << "[" << currentTest << "/" << totalTests << "] "
                     << filterNames[filterIdx] << " | " << modeNames[modeIdx]
                     << " | " << res.name << endl;

                vector<double> frameTimes;

                for (int frame = 0; frame < BENCHMARK_FRAMES; frame++)
                {
                    auto frameStart = chrono::high_resolution_clock::now();

                    Mat capturedFrame;
                    cap >> capturedFrame;
                    if (capturedFrame.empty())
                    {
                        cerr << "Error: Could not read frame during benchmark!" << endl;
                        return;
                    }

                    Mat displayFrame;
                    unsigned int shaderToUse = shaders[0];

                    if (useGPU)
                    {
                        displayFrame = capturedFrame;
                        shaderToUse = shaders[filterIdx];
                    }
                    else
                    {
                        // Applying CPU filter based on filter index
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
                        displayFrame = applyCPUTransform(displayFrame, noTransform);
                    }

                    // Converting BGR to RGB for OpenGL
                    Mat frame_rgb;
                    cvtColor(displayFrame, frame_rgb, COLOR_BGR2RGB);

                    // Uploading texture to GPU
                    glBindTexture(GL_TEXTURE_2D, texture);
                    // Upload the texture data
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.cols, frame_rgb.rows,
                                 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.data);

                    // Clearing the screen and using the appropriate shader
                    glClear(GL_COLOR_BUFFER_BIT);
                    // Using the selected shader program
                    glUseProgram(shaderToUse);

                    // Setting transform uniforms if using GPU
                    if (useGPU)
                    {
                        GLint translateLoc = glGetUniformLocation(shaderToUse, "uTranslate");
                        GLint scaleLoc = glGetUniformLocation(shaderToUse, "uScale");
                        GLint rotationLoc = glGetUniformLocation(shaderToUse, "uRotation");

                        glUniform2f(translateLoc, noTransform.translateX, noTransform.translateY);
                        glUniform1f(scaleLoc, noTransform.scale);
                        glUniform1f(rotationLoc, noTransform.rotation * 3.14159265f / 180.0f);
                    }

                    // Setting additional uniforms for specific filters
                    if (filterIdx == 2 || filterIdx == 3 || filterIdx == 5)
                    {
                        GLint texelLoc = glGetUniformLocation(shaderToUse, "texelSize");
                        glUniform2f(texelLoc, 1.0f / frame_rgb.cols, 1.0f / frame_rgb.rows);
                    }

                    // Setting pixelation uniforms
                    if (filterIdx == 4)
                    {
                        GLint pixelLoc = glGetUniformLocation(shaderToUse, "pixelSize");
                        GLint resLoc = glGetUniformLocation(shaderToUse, "resolution");
                        glUniform1f(pixelLoc, 10.0f);
                        glUniform2f(resLoc, (float)frame_rgb.cols, (float)frame_rgb.rows);
                    }

                    // Drawing the frame
                    glBindVertexArray(VAO);
                    // Drawing the elements
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                    // Swapping buffers and polling events
                    glfwSwapBuffers(window);
                    // Poll for and process events
                    glfwPollEvents();

                    auto frameEnd = chrono::high_resolution_clock::now();
                    double frameTime = chrono::duration<double, milli>(frameEnd - frameStart).count();
                    frameTimes.push_back(frameTime);

                    if ((frame + 1) % 25 == 0)
                    {
                        cout << "    Progress: " << (frame + 1) << "/" << BENCHMARK_FRAMES << " frames" << endl;
                    }
                }

                double avgFrameTime = 0.0;
                for (double ft : frameTimes)
                {
                    avgFrameTime += ft;
                }
                avgFrameTime /= frameTimes.size();
                double avgFPS = 1000.0 / avgFrameTime;

                // Logging performance data
                PerformanceData data;
                data.filterName = filterNames[filterIdx];
                data.executionMode = modeNames[modeIdx];
                data.resolution = actualWidth * actualHeight;
                data.hasTransform = false;
                data.fps = avgFPS;
                data.frameTime = avgFrameTime;
                performanceLog.push_back(data);

                cout << "avg FPS: " << fixed << setprecision(2) << avgFPS
                     << " | Frame Time: " << avgFrameTime << "ms\n"
                     << endl;
            }
        }
    }

    // Restoring original resolution
    cap.set(CAP_PROP_FRAME_WIDTH, originalWidth);
    cap.set(CAP_PROP_FRAME_HEIGHT, originalHeight);
    glfwSetWindowSize(window, originalWidth, originalHeight);

    // Restoring viewport
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);

    cout << "Done with resolution testing" << endl;
    cout << "Total measurements: " << performanceLog.size() << endl;
    cout << "Resolution restored to: " << originalWidth << "x" << originalHeight << endl;

    writePerformanceCSV(performanceLog, "performance_resolution.csv");
    cout << "Results saved to performance_resolution.csv\n"
         << endl;
}

void runAutomatedBenchmark(
    // Video capture object
    VideoCapture &cap,
    // OpenGL window
    GLFWwindow *window,
    // Shader programs
    unsigned int *shaders,
    // Vertex Array Object
    unsigned int VAO,
    // Texture
    unsigned int texture,
    // Performance log
    vector<PerformanceData> &performanceLog,
    // Whether to test with transforms
    bool testWithTransforms)
{
    // Benchmarking logic, same as before
    const int BENCHMARK_FRAMES = 50;
    const char *filterNames[] = {"None", "Grayscale", "Blur", "Edge", "Pixelation", "Comic"};
    const int numFilters = 6;
    const bool executionModes[] = {true, false};
    const char *modeNames[] = {"GPU", "CPU"};

    // Setting up transform configurations
    vector<TransformParams> transformConfigs;
    vector<string> transformNames;

    // Initial no transform
    TransformParams noTransform;
    noTransform.translateX = 0.0f;
    noTransform.translateY = 0.0f;
    noTransform.scale = 1.0f;
    noTransform.rotation = 0.0f;
    transformConfigs.push_back(noTransform);
    transformNames.push_back("No Transform");

    // Additional transforms if enabled
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

    cout << "Starting the automated transform benchmarking" << endl;
    cout << "Total tests: " << totalTests << endl;
    cout << "Frames per test: " << BENCHMARK_FRAMES << endl;
    cout << "Transform configurations: " << transformConfigs.size() << endl;

    // Going through each transform configuration
    for (size_t transformIdx = 0; transformIdx < transformConfigs.size(); transformIdx++)
    {
        // Current transform and its name
        const TransformParams &currentTransform = transformConfigs[transformIdx];
        // Name of the current transform
        const string &transformName = transformNames[transformIdx];

        cout << "Testing Transform Config: " << transformName << endl;
        cout << "  Translation: (" << currentTransform.translateX << ", " << currentTransform.translateY << ")" << endl;
        cout << "  Scale: " << currentTransform.scale << endl;
        cout << "  Rotation: " << currentTransform.rotation << "°\n"
             << endl;

        // Going through each execution mode
        for (int modeIdx = 0; modeIdx < 2; modeIdx++)
        {
            bool useGPU = executionModes[modeIdx];

            // Going through each filter
            for (int filterIdx = 0; filterIdx < numFilters; filterIdx++)
            {
                currentTest++;
                cout << "[" << currentTest << "/" << totalTests << "] "
                     << filterNames[filterIdx] << " | " << modeNames[modeIdx]
                     << " | " << transformName << endl;

                vector<double> frameTimes;

                for (int frame = 0; frame < BENCHMARK_FRAMES; frame++)
                {
                    auto frameStart = chrono::high_resolution_clock::now();

                    Mat capturedFrame;
                    cap >> capturedFrame;
                    if (capturedFrame.empty())
                    {
                        cerr << "Error: Could not read frame during benchmark!" << endl;
                        return;
                    }
                    // Frame processing
                    Mat displayFrame;
                    // Determine the shader to use
                    unsigned int shaderToUse = shaders[0];

                    if (useGPU)
                    {
                        displayFrame = capturedFrame;
                        shaderToUse = shaders[filterIdx];
                    }
                    else
                    {
                        // Applying CPU filter based on filter index
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
                        displayFrame = applyCPUTransform(displayFrame, currentTransform);
                    }

                    // Converting BGR to RGB for OpenGL
                    Mat frame_rgb;
                    cvtColor(displayFrame, frame_rgb, COLOR_BGR2RGB);

                    // Uploading texture to GPU
                    glBindTexture(GL_TEXTURE_2D, texture);
                    // Upload the texture data
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.cols, frame_rgb.rows,
                                 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.data);

                    // Clearing the screen and using the appropriate shader
                    glClear(GL_COLOR_BUFFER_BIT);
                    // Using the selected shader program
                    glUseProgram(shaderToUse);

                    // Setting transform uniforms if using GPU
                    if (useGPU)
                    {
                        // Setting transform uniforms
                        GLint translateLoc = glGetUniformLocation(shaderToUse, "uTranslate");
                        // Setting translation values
                        GLint scaleLoc = glGetUniformLocation(shaderToUse, "uScale");
                        // Setting scale values
                        GLint rotationLoc = glGetUniformLocation(shaderToUse, "uRotation");

                        // Uploading transform values to shader
                        glUniform2f(translateLoc, currentTransform.translateX, currentTransform.translateY);
                        // Uploading scale value
                        glUniform1f(scaleLoc, currentTransform.scale);
                        // Uploading rotation value in radians
                        glUniform1f(rotationLoc, currentTransform.rotation * 3.14159265f / 180.0f);
                    }

                    // Setting additional uniforms for specific filters
                    if (filterIdx == 2 || filterIdx == 3 || filterIdx == 5)
                    {
                        // Setting texel size uniform
                        GLint texelLoc = glGetUniformLocation(shaderToUse, "texelSize");
                        // Uploading texel size
                        glUniform2f(texelLoc, 1.0f / frame_rgb.cols, 1.0f / frame_rgb.rows);
                    }

                    // Setting pixelation uniforms
                    if (filterIdx == 4)
                    {
                        // Setting pixelation size uniform
                        GLint pixelLoc = glGetUniformLocation(shaderToUse, "pixelSize");
                        // Setting resolution uniform
                        GLint resLoc = glGetUniformLocation(shaderToUse, "resolution");
                        // Uploading pixel size
                        glUniform1f(pixelLoc, 10.0f);
                        // Uploading resolution
                        glUniform2f(resLoc, (float)frame_rgb.cols, (float)frame_rgb.rows);
                    }

                    // Drawing the frame
                    glBindVertexArray(VAO);
                    // Drawing the elements
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

                    // Swapping buffers and polling events
                    glfwSwapBuffers(window);
                    // Poll for and process events
                    glfwPollEvents();

                    auto frameEnd = chrono::high_resolution_clock::now();
                    double frameTime = chrono::duration<double, milli>(frameEnd - frameStart).count();
                    frameTimes.push_back(frameTime);

                    if ((frame + 1) % 50 == 0)
                    {
                        cout << "    Progress: " << (frame + 1) << "/" << BENCHMARK_FRAMES << " frames" << endl;
                    }
                }

                double avgFrameTime = 0.0;
                for (double ft : frameTimes)
                {
                    avgFrameTime += ft;
                }
                avgFrameTime /= frameTimes.size();
                double avgFPS = 1000.0 / avgFrameTime;

                bool hasTransform = (transformIdx > 0);

                // Logging performance data
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

    cout << "Done with testing transforms" << endl;
    cout << "Total measurements: " << performanceLog.size() << endl;

    writePerformanceCSV(performanceLog, "performance_transforms.csv");
    cout << "saved to performance_transforms.csv" << endl;
}