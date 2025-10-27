#ifndef INPUT_HANDLER_H
#define INPUT_HANDLER_H

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "types.h"

class InputHandler
{
public:
    // Setup all input callbacks
    static void setupCallbacks(GLFWwindow *window, MouseData *data);

    // Individual callback functions
    static void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
    static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
};

#endif