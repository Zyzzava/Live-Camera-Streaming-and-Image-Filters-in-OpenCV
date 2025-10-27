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
    const bool executionModes[] = {true, false};
    const char *modeNames[] = {"GPU", "CPU"};

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

    int originalWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    int originalHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

    int totalTests = numFilters * 2 * resolutions.size();
    int currentTest = 0;

    cout << "Resolution benchmark" << endl;
    cout << "Total tests: " << totalTests << endl;
    cout << "Frames per test: " << BENCHMARK_FRAMES << endl;
    cout << "Resolutions to test: " << resolutions.size() << endl;

    TransformParams noTransform;
    noTransform.translateX = 0.0f;
    noTransform.translateY = 0.0f;
    noTransform.scale = 1.0f;
    noTransform.rotation = 0.0f;

    for (const auto &res : resolutions)
    {
        cout << "Testing Resolution: " << res.name << endl;

        cap.set(CAP_PROP_FRAME_WIDTH, res.width);
        cap.set(CAP_PROP_FRAME_HEIGHT, res.height);

        int actualWidth = cap.get(CAP_PROP_FRAME_WIDTH);
        int actualHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
        cout << "Actual resolution: " << actualWidth << "x" << actualHeight << endl;

        glfwSetWindowSize(window, actualWidth, actualHeight);

        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);

        for (int modeIdx = 0; modeIdx < 2; modeIdx++)
        {
            bool useGPU = executionModes[modeIdx];

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

                    Mat frame_rgb;
                    cvtColor(displayFrame, frame_rgb, COLOR_BGR2RGB);

                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.cols, frame_rgb.rows,
                                 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.data);

                    glClear(GL_COLOR_BUFFER_BIT);
                    glUseProgram(shaderToUse);

                    if (useGPU)
                    {
                        GLint translateLoc = glGetUniformLocation(shaderToUse, "uTranslate");
                        GLint scaleLoc = glGetUniformLocation(shaderToUse, "uScale");
                        GLint rotationLoc = glGetUniformLocation(shaderToUse, "uRotation");

                        glUniform2f(translateLoc, noTransform.translateX, noTransform.translateY);
                        glUniform1f(scaleLoc, noTransform.scale);
                        glUniform1f(rotationLoc, noTransform.rotation * 3.14159265f / 180.0f);
                    }

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

    cap.set(CAP_PROP_FRAME_WIDTH, originalWidth);
    cap.set(CAP_PROP_FRAME_HEIGHT, originalHeight);
    glfwSetWindowSize(window, originalWidth, originalHeight);

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
    VideoCapture &cap,
    GLFWwindow *window,
    unsigned int *shaders,
    unsigned int VAO,
    unsigned int texture,
    vector<PerformanceData> &performanceLog,
    bool testWithTransforms)
{
    const int BENCHMARK_FRAMES = 50;
    const char *filterNames[] = {"None", "Grayscale", "Blur", "Edge", "Pixelation", "Comic"};
    const int numFilters = 6;
    const bool executionModes[] = {true, false};
    const char *modeNames[] = {"GPU", "CPU"};

    vector<TransformParams> transformConfigs;
    vector<string> transformNames;

    TransformParams noTransform;
    noTransform.translateX = 0.0f;
    noTransform.translateY = 0.0f;
    noTransform.scale = 1.0f;
    noTransform.rotation = 0.0f;
    transformConfigs.push_back(noTransform);
    transformNames.push_back("No Transform");

    if (testWithTransforms)
    {
        TransformParams translateOnly;
        translateOnly.translateX = 0.3f;
        translateOnly.translateY = 0.2f;
        translateOnly.scale = 1.0f;
        translateOnly.rotation = 0.0f;
        transformConfigs.push_back(translateOnly);
        transformNames.push_back("Translation");

        TransformParams scaleOnly;
        scaleOnly.translateX = 0.0f;
        scaleOnly.translateY = 0.0f;
        scaleOnly.scale = 1.5f;
        scaleOnly.rotation = 0.0f;
        transformConfigs.push_back(scaleOnly);
        transformNames.push_back("Scale");

        TransformParams rotateOnly;
        rotateOnly.translateX = 0.0f;
        rotateOnly.translateY = 0.0f;
        rotateOnly.scale = 1.0f;
        rotateOnly.rotation = 25.0f;
        transformConfigs.push_back(rotateOnly);
        transformNames.push_back("Rotation");

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

    for (size_t transformIdx = 0; transformIdx < transformConfigs.size(); transformIdx++)
    {
        const TransformParams &currentTransform = transformConfigs[transformIdx];
        const string &transformName = transformNames[transformIdx];

        cout << "Testing Transform Config: " << transformName << endl;
        cout << "  Translation: (" << currentTransform.translateX << ", " << currentTransform.translateY << ")" << endl;
        cout << "  Scale: " << currentTransform.scale << endl;
        cout << "  Rotation: " << currentTransform.rotation << "°\n"
             << endl;

        for (int modeIdx = 0; modeIdx < 2; modeIdx++)
        {
            bool useGPU = executionModes[modeIdx];

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

                    Mat displayFrame;
                    unsigned int shaderToUse = shaders[0];

                    if (useGPU)
                    {
                        displayFrame = capturedFrame;
                        shaderToUse = shaders[filterIdx];
                    }
                    else
                    {
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

                    Mat frame_rgb;
                    cvtColor(displayFrame, frame_rgb, COLOR_BGR2RGB);

                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.cols, frame_rgb.rows,
                                 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.data);

                    glClear(GL_COLOR_BUFFER_BIT);
                    glUseProgram(shaderToUse);

                    if (useGPU)
                    {
                        GLint translateLoc = glGetUniformLocation(shaderToUse, "uTranslate");
                        GLint scaleLoc = glGetUniformLocation(shaderToUse, "uScale");
                        GLint rotationLoc = glGetUniformLocation(shaderToUse, "uRotation");

                        glUniform2f(translateLoc, currentTransform.translateX, currentTransform.translateY);
                        glUniform1f(scaleLoc, currentTransform.scale);
                        glUniform1f(rotationLoc, currentTransform.rotation * 3.14159265f / 180.0f);
                    }

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