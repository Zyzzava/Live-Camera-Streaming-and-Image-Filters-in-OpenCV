#include <iostream>
#include <opencv2/opencv.hpp>
#include "filters.h"

using namespace std;
using namespace cv;

// Driver Code
int main()
{
    // Create VideoCapture object
    VideoCapture cap(1);
    // Check if camera opened successfully
    if (!cap.isOpened())
    {
        cerr << "Error: Could not open camera!" << endl;
        return -1;
    }
    cout << "Camera opened successfully! Press keys to apply filters:" << endl;
    cout << "1 - Grayscale, 2 - Gaussian Blur, 3 - Edge Detection" << endl;
    cout << "SPACE - Original, 'q' - Quit" << endl;
    // Give the camera some time to initialize
    cout << "Waiting for camera to initialize..." << endl;
    waitKey(2000); // Wait 2 seconds

    Mat frame;
    int attempts = 0;
    const int max_attempts = 10;
    int currentFilter = 0; // 0 = original, 1-3 = different filters

    // Main loop to capture and display frames
    while (true)
    {
        // Capture frame-by-frame
        cap >> frame;

        // Check if frame is empty
        if (frame.empty())
        {
            attempts++;
            if (attempts >= max_attempts)
            {
                cerr << "Error: Could not read frame from camera after " << max_attempts << " attempts!" << endl;
                cerr << "This might be a camera permission issue on macOS." << endl;
                cerr << "Please check System Settings > Privacy & Security > Camera and enable Terminal." << endl;
                break;
            }
            cout << "Attempt " << attempts << "/" << max_attempts << " - waiting for camera..." << endl;
            waitKey(500); // Wait 500ms before next attempt
            continue;
        }
        // Reset attempts counter if we successfully got a frame
        attempts = 0;

        // Apply selected filter
        Mat displayFrame = frame.clone();
        switch (currentFilter)
        {
        case 1:
            displayFrame = ImageFilters::applyGrayscale(frame);
            break;
        case 2:
            displayFrame = ImageFilters::applyGaussianBlur(frame, 15);
            break;
        case 3:
            displayFrame = ImageFilters::applyEdgeDetection(frame);
            break;
        default:
            // Original frame
            break;
        }

        // Display the frame
        imshow("Live Camera Feed with Filters", displayFrame);

        // Check for key press and break loop if 'q' is pressed
        char key = waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q')
        {
            cout << "Exiting..." << endl;
            break;
        }

        // Filter selection
        switch (key)
        {
        case '1':
            currentFilter = 1;
            cout << "Applied: Grayscale" << endl;
            break;
        case '2':
            currentFilter = 2;
            cout << "Applied: Gaussian Blur" << endl;
            break;
        case '3':
            currentFilter = 3;
            cout << "Applied: Edge Detection" << endl;
            break;
        case ' ': // Add this case for space
            currentFilter = 0;
            cout << "Applied: Original" << endl;
            break;
        }
    }
    // Release the camera and close windows
    cap.release();
    destroyAllWindows();

    return 0;
}