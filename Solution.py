import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial import distance
from filterpy.kalman import KalmanFilter

class OptimizedObjectTracker:
    def __init__(self, max_objects=3, min_object_size=1500, max_disappeared=30):
        self.object_detector = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=40, 
            detectShadows=True
        )
        self.tracked_objects = {}
        self.disappeared_objects = {}
        self.next_object_id = 0

        self.max_objects = max_objects
        self.min_object_size = min_object_size
        self.max_disappeared = max_disappeared

    def detect_objects(self, frame, depth_frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self.object_detector.apply(blurred)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_object_size:
                x, y, w, h = cv2.boundingRect(cnt)
                
                depth_roi = depth_frame[y:y+h, x:x+w]
                avg_depth = np.mean(depth_roi[depth_roi > 0]) / 1000 if depth_roi.size > 0 else 0
                
                centroid = (x + w // 2, y + h // 2)
                detections.append({
                    'bbox': (x, y, w, h),
                    'centroid': centroid,
                    'area': area,
                    'depth': avg_depth
                })

        return sorted(detections, key=lambda x: x['area'], reverse=True)[:self.max_objects]

    def update_tracks(self, detections):

        if not self.tracked_objects:
            for detection in detections:
                self._register_object(detection)
            return

        matched_indices = []
        unmatched_detections = []

        for idx, detection in enumerate(detections):
            best_match = None
            min_distance = float('inf')

            for track_id, track in self.tracked_objects.items():
                dist = distance.euclidean(detection['centroid'], track['centroid'])
                if dist < min_distance and dist < 100:  # Distance threshold
                    min_distance = dist
                    best_match = track_id

            if best_match is not None:
                matched_indices.append((best_match, idx))
            else:
                unmatched_detections.append(detection)

        for track_id, det_idx in matched_indices:
            self._update_track(track_id, detections[det_idx])
        for detection in unmatched_detections:
            self._register_object(detection)

        
        self._prune_disappeared_tracks()

    def _register_object(self, detection):
        track_id = self.next_object_id
        self.tracked_objects[track_id] = {
            'centroid': detection['centroid'],
            'bbox': detection['bbox'],
            'depth': detection['depth'],
            'history': [detection['centroid']]
        }
        self.disappeared_objects[track_id] = 0
        self.next_object_id += 1

    def _update_track(self, track_id, detection):
        track = self.tracked_objects[track_id]
        track['centroid'] = detection['centroid']
        track['bbox'] = detection['bbox']
        track['depth'] = detection['depth']
        track['history'].append(detection['centroid'])
        
        
        track['history'] = track['history'][-10:]
        self.disappeared_objects[track_id] = 0

    def _prune_disappeared_tracks(self):
        tracks_to_delete = []
        for track_id in list(self.disappeared_objects.keys()):
            self.disappeared_objects[track_id] += 1
            
            if self.disappeared_objects[track_id] > self.max_disappeared:
                tracks_to_delete.append(track_id)

        for track_id in tracks_to_delete:
            del self.tracked_objects[track_id]
            del self.disappeared_objects[track_id]

def process_realsense_tracking():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
    tracker = OptimizedObjectTracker()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            detections = tracker.detect_objects(color_image, depth_image)
            tracker.update_tracks(detections)

            
            for track_id, track in tracker.tracked_objects.items():
                x, y, w, h = track['bbox']
                cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(color_image, 
                    f'ID:{track_id} Depth:{track["depth"]:.2f}m', 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            cv2.imshow('Optimized Object Tracking', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_realsense_tracking()