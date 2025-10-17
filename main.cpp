#include <iostream>
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

    void main() {
        // Apply scale then translation
        vec2 pos = aPos * uScale + uTranslate;
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
};

// Mouse callback data
struct MouseData
{
    double lastX = 0.0;
    double lastY = 0.0;
    bool isDragging = false;
    TransformParams transform;
};

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

    // Create transformation matrix
    // Scale matrix
    Mat scaleMat = (Mat_<float>(2, 3) << params.scale, 0, 0,
                    0, params.scale, 0);

    // Translation in pixels (convert from normalized -1 to 1 to pixel coords)
    float txPixels = params.translateX * input.cols / 2.0f;
    float tyPixels = -params.translateY * input.rows / 2.0f; // Flip Y back

    // Combined transformation: scale around center, then translate
    Point2f center(input.cols / 2.0f, input.rows / 2.0f);

    // First translate to origin, scale, translate back, then apply user translation
    Mat transform = getRotationMatrix2D(center, 0, params.scale);
    transform.at<double>(0, 2) += txPixels;
    transform.at<double>(1, 2) += tyPixels;

    warpAffine(input, output, transform, input.size(), INTER_LINEAR);

    return output;
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

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
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

        // Use the shader program to draw
        glUseProgram(shaderToUse);

        // Set transformation uniforms (GPU mode)
        if (useGPU)
        {
            GLint translateLoc = glGetUniformLocation(shaderToUse, "uTranslate");
            GLint scaleLoc = glGetUniformLocation(shaderToUse, "uScale");
            glUniform2f(translateLoc, mouseData.transform.translateX, mouseData.transform.translateY);
            glUniform1f(scaleLoc, mouseData.transform.scale);
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
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        {
            // Reset transformation
            mouseData.transform.translateX = 0.0f;
            mouseData.transform.translateY = 0.0f;
            mouseData.transform.scale = 1.0f;
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