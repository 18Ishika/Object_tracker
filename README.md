OBJECT DETECTION AND TRACKER

This project implements an optimized object detection and tracking system using Intel RealSense cameras, OpenCV, and Kalman filters. It tracks objects in real-time, provides depth information from RealSense frames, and handles object disappearance with optimization techniques.

Features

1. Real-time Object Detection using background subtraction and contour detection.

2. Object Tracking: Tracks objects using their centroid, bounding box, and depth.

3. Depth Mapping with Intel RealSense for precise object distance measurement.

4. Optimized Tracking: Supports up to three objects, handles object disappearance efficiently.

5. Visualization: Displays object tracking with bounding boxes, IDs, and depth information.


Tech Stack

1. Python: The primary programming language used.

2. OpenCV: For image processing and object detection./n

3. Intel RealSense SDK (pyrealsense2): For capturing depth and RGB video frames from the Intel RealSense camera.

4. Scipy: Used for calculating Euclidean distances between centroids.

5. FilterPy: Kalman filters for object tracking (planned future enhancement).

6. NumPy: For efficient numerical operations and array handling.


Intel IPP (Integrated Performance Primitives)

OpenCV uses Intel IPP for accelerated image processing, enabling faster execution of critical operations like filtering, blurring, and contour detection. IPP optimizes performance on Intel processors by utilizing multi-core parallelism and SIMD (Single Instruction Multiple Data) instructions. This helps achieve real-time object detection and tracking even in computationally intensive tasks.

Performance Benefits
Faster Execution: Intel IPP accelerates fundamental operations like Gaussian blurring (cv2.GaussianBlur()) and morphological transformations (cv2.morphologyEx()), improving performance, especially on Intel processors.

Optimized Multi-core Processing: IPP efficiently uses multiple cores and SIMD instructions for faster data processing.

Cross-platform Support: While optimized for Intel architecture, Intel IPP enhances OpenCV on any platform without compromising performance on non-Intel systems.



PROJECT WORKFLOW

1. Initialization:

    The project initializes the Intel RealSense camera to capture depth and RGB frames and sets up OpenCV background subtraction for object detection.

2. Object Detection:

    Preprocessing: Convert to grayscale, apply Gaussian blur, and create a mask using background subtraction.

    Contours: Find contours of detected objects and filter based on size.

    Depth Estimation: Calculate the average depth of each detected object using the RealSense depth frame.

3. Object Tracking:

    Match detected objects with previously tracked ones using centroid distance.

    Update tracks for matched objects and register new ones for unmatched detections.

    Handle object disappearance and pruning of outdated tracks.

4. Visualization:

    Draw bounding boxes and centroids around detected objects.

    Display object IDs and corresponding depth information.

5. Termination:

    The pipeline is stopped, and OpenCV windows are closed when the user presses q.


PREQUISITES

Hardware Requirements
    Intel RealSense Camera (e.g., D435, D455).
    A computer with Intel processors for maximum performance benefits using Intel IPP in OpenCV.

Software Requirements
    Python 3.7+
    Intel RealSense SDK (pyrealsense2)
    OpenCV
    NumPy
    Scipy
    FilterPy (optional, for Kalman filter implementation)
    Intel IPP (via OpenCV)


Installation

Clone the repository:

    git clone https://github.com/yourusername/OptimizedObjectTracker.git
    cd OptimizedObjectTracker
   
Create a virtual environment:

    python -m venv env
    source env/bin/activate  # For Linux/Mac
    env\Scripts\activate     # For Windows
    
Install depencies

     pip install opencv-python-headless numpy scipy filterpy pyrealsense2
     
Connect the Intel RealSense camera and ensure the RealSense SDK is correctly installed.
   
Usage

Run the main script:
      python main.py
The program will start capturing frames from the RealSense camera. It will display a real-time feed with bounding boxes around detected objects and their IDs and depth information.
Press q to exit the program.

ObjectTracker

├── Solution.py         # script that runs the object tracker.

├── README.md       # Project documentation (this file).


Output 

When running the project:
    
    python Solution.py

A real-time video feed will be shown.
Detected objects will be outlined with bounding boxes.
Object IDs and depth information will be displayed on the screen.
