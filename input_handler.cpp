#include "input_handler.h"
#include <algorithm>

// Setup all input callbacks
void InputHandler::setupCallbacks(GLFWwindow *window, MouseData *data)
{
    // Setting user pointer to access MouseData in callbacks
    glfwSetWindowUserPointer(window, data);
    // Setting up the callbacks
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    // Setting cursor position callback
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    // Setting scroll callback
    glfwSetScrollCallback(window, scrollCallback);
}

// Mouse button callback implementation
void InputHandler::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    // Retrieving MouseData from window user pointer
    MouseData *data = static_cast<MouseData *>(glfwGetWindowUserPointer(window));

    // Handling left mouse button for dragging
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        // On press, start dragging and record position
        if (action == GLFW_PRESS)
        {
            data->isDragging = true;
            // Get current cursor position
            glfwGetCursorPos(window, &data->lastX, &data->lastY);
        }
        // On release, stop dragging
        else if (action == GLFW_RELEASE)
        {
            // Stop dragging
            data->isDragging = false;
        }
    }
    // Handling right mouse button for rotating
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        // On press, start rotating and record position
        if (action == GLFW_PRESS)
        {
            data->isRotating = true;
            glfwGetCursorPos(window, &data->lastX, &data->lastY);
        }
        // On release, stop rotating
        else if (action == GLFW_RELEASE)
        {
            data->isRotating = false;
        }
    }
}

// Cursor position callback implementation
void InputHandler::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    // Retrieving MouseData from window user pointer
    MouseData *data = static_cast<MouseData *>(glfwGetWindowUserPointer(window));

    // If dragging, update translation based on mouse movement
    if (data->isDragging)
    {
        double dx = xpos - data->lastX;
        double dy = ypos - data->lastY;

        int width, height;
        // Getting window size for normalization
        glfwGetWindowSize(window, &width, &height);

        // Convert pixel movement to normalized coordinates (-1 to 1)
        data->transform.translateX += (float)(dx / width) * 2.0f;
        // Flip Y axis for correct direction
        data->transform.translateY -= (float)(dy / height) * 2.0f; // Flip Y

        // Update last positions
        data->lastX = xpos;
        data->lastY = ypos;
    }
    // If rotating, update rotation based on horizontal mouse movement
    else if (data->isRotating)
    {
        double dx = xpos - data->lastX;

        // Horizontal mouse movement controls rotation
        // Sensitivity: 0.5 degrees per pixel
        data->transform.rotation += (float)dx * 0.5f;

        // Update last positions
        data->lastX = xpos;
        data->lastY = ypos;
    }
}

// Scroll callback implementation
void InputHandler::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    // Retrieving MouseData from window user pointer
    MouseData *data = static_cast<MouseData *>(glfwGetWindowUserPointer(window));
    // Zooming in/out based on scroll input
    data->transform.scale += (float)yoffset * 0.1f;
    // Clamp scale between 0.1 and 5.0
    data->transform.scale = std::max(0.1f, std::min(data->transform.scale, 5.0f));
}