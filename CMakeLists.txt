cmake_minimum_required(VERSION 3.10)
project(OpenGLWebcam)

set(CMAKE_CXX_STANDARD 14)

# Explicitly specify the path to the GLEW library
set(GLEW_LIBRARY "/opt/homebrew/lib/libGLEW.dylib" CACHE STRING "Path to GLEW library" FORCE)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

if (GLEW_FOUND)
    message(STATUS "GLEW found: ${GLEW_LIBRARIES}")
else()
    message(FATAL_ERROR "GLEW not found")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/filters)

add_executable(OpenGLWebcam main.cpp filters/filters.cpp filters/filters.h)

target_link_libraries(OpenGLWebcam ${OpenCV_LIBS} OpenGL::GL glfw ${GLEW_LIBRARY})