import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.ndimage import gaussian_filter1d

class ObjectTracker:
    def __init__(self):
        self.object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)
        self.tracked_objects = {}
        self.next_object_id = 0
        self.max_objects = 3  # Maximum number of objects to track
        self.thickness = 3    # Path line thickness
        self.trail_length = 15  # Maximum number of points in the trail

    def detect_objects(self, frame):
        """Detect objects in the given frame using background subtraction and contour detection."""
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Pre-smoothing for noise reduction
        mask = self.object_detector.apply(blurred_frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1500:  # Threshold for larger objects
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h, area])

        # Sort by area (largest first) and select top N objects
        detections = sorted(detections, key=lambda x: x[4], reverse=True)[:self.max_objects]
        return detections

    def smooth_path(self, positions):
        """Smooth the path of an object using Gaussian filtering."""
        if len(positions) < 3:
            return positions
        x_coords = np.array([pos[0] for pos in positions], dtype=np.float32)
        y_coords = np.array([pos[1] for pos in positions], dtype=np.float32)

        # Apply Gaussian filter for smoother path
        smoothed_x = gaussian_filter1d(x_coords, sigma=1)
        smoothed_y = gaussian_filter1d(y_coords, sigma=1)

        return list(zip(smoothed_x.astype(int), smoothed_y.astype(int)))

    def update_tracked_objects(self, detections):
        """Update the tracked objects based on new detections."""
        centroids = [(x + w // 2, y + h // 2) for x, y, w, h, _ in detections]

        new_tracked_objects = {}
        for object_id, positions in self.tracked_objects.items():
            if centroids:
                # Match existing object to the closest new centroid
                distances = [np.linalg.norm(np.array(centroid) - np.array(positions[-1])) for centroid in centroids]
                min_distance_idx = np.argmin(distances)
                if distances[min_distance_idx] < 50:  # Threshold for linking objects
                    new_tracked_objects[object_id] = positions + [centroids[min_distance_idx]]
                    centroids.pop(min_distance_idx)

        # Add new objects for unmatched centroids
        for centroid in centroids:
            new_tracked_objects[self.next_object_id] = [centroid]
            self.next_object_id += 1

        # Trim trail length
        for object_id in new_tracked_objects:
            if len(new_tracked_objects[object_id]) > self.trail_length:
                new_tracked_objects[object_id] = new_tracked_objects[object_id][-self.trail_length:]

        self.tracked_objects = new_tracked_objects

    def draw_tracked_objects(self, frame):
        """Draw the tracked objects and their trails on the frame."""
        for object_id, positions in self.tracked_objects.items():
            if len(positions) < 2:
                continue

            # Smooth trail with Gaussian smoothing
            smoothed_positions = self.smooth_path(positions)

            # Draw trail following the object
            for i in range(1, len(smoothed_positions)):
                cv2.line(frame, smoothed_positions[i - 1], smoothed_positions[i], (0, 255, 255), self.thickness)

            # Highlight the current position
            x, y = smoothed_positions[-1]
            cv2.putText(frame, f'ID {object_id}', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)

    def get_distance(self, depth_frame, x, y):
        """Retrieve the distance of the object at (x, y) from the depth frame."""
        if depth_frame is None:
            return None

        try:
            # Ensure coordinates are within the frame bounds
            height, width = depth_frame.shape
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)

            # Return the distance at the given point
            return depth_frame[y, x]
        except Exception as e:
            print(f"Error retrieving depth: {e}")
            return None


def process_realsense_stream():
    tracker = ObjectTracker()

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        pipeline.start(config)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Detect objects
            detections = tracker.detect_objects(color_image)
            tracker.update_tracked_objects(detections)

            # Draw bounding boxes and depth info
            for box in detections:
                x, y, w, h, _ = box
                distance = tracker.get_distance(depth_image, x + w // 2, y + h // 2)

                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f'Dist: {distance / 1000:.2f}m' if distance else 'Dist: N/A'
                cv2.putText(color_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            tracker.draw_tracked_objects(color_image)

            # Display the frame
            cv2.imshow('Object Tracker with RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop RealSense pipeline
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_realsense_stream()
