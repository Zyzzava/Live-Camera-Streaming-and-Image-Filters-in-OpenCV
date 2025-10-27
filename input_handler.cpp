#include "input_handler.h"
#include <algorithm>

void InputHandler::setupCallbacks(GLFWwindow *window, MouseData *data)
{
    glfwSetWindowUserPointer(window, data);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetScrollCallback(window, scrollCallback);
}

void InputHandler::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
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

void InputHandler::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
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

void InputHandler::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    MouseData *data = static_cast<MouseData *>(glfwGetWindowUserPointer(window));
    data->transform.scale += (float)yoffset * 0.1f;
    // Clamp scale between 0.1 and 5.0
    data->transform.scale = std::max(0.1f, std::min(data->transform.scale, 5.0f));
}