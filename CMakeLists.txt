cmake_minimum_required ( VERSION 3.10)
project ( RealtimeVP )
find_package ( OpenCV REQUIRED )

# Include filters directory
include_directories(filters)

# Add all source files
add_executable ( RealtimeVP 
    main.cpp 
    filters/filters.cpp
)

target_link_libraries ( RealtimeVP ${OpenCV_LIBS} )