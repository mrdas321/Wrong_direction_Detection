import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

class WrongDirectionDetector:
    def __init__(self, video_path, output_path="output_violation.mp4"):
        self.video_path = video_path
        
        
        # Checks if NVIDIA GPU is available, otherwise defaults to CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model and move it to the selected device
        self.model = YOLO("yolov8n.pt").to(self.device)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(" Cannot open video")

        # --- VIDEO EXPORT SETUP ---
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        os.makedirs("violations", exist_ok=True)

        self.divider = np.array([[431, 714], [804, 716], [621, 282], [608, 282]])
        self.track_history = {}
        self.id_to_name = {}
        self.counts = {"person": 0, "car": 0, "bike": 0, "bus": 0, "truck": 0}
        self.class_map = {0: "person", 2: "car", 3: "bike", 5: "bus", 7: "truck"}

    def get_lane(self, cx, cy, height):
        x_top = (self.divider[2][0] + self.divider[3][0]) // 2
        x_bottom = (self.divider[0][0] + self.divider[1][0]) // 2
        divider_x = int(x_top + (cy / height) * (x_bottom - x_top))
        return "RIGHT" if cx > divider_x else "LEFT"

    def get_name(self, track_id, label):
        if track_id not in self.id_to_name:
            self.counts[label] += 1
            self.id_to_name[track_id] = f"{label}{self.counts[label]}"
        return self.id_to_name[track_id]

    def is_wrong_direction(self, track_id, lane):
        history = self.track_history[track_id]
        if len(history) < 5: return False
        dy = history[-1][1] - history[0][1]
        if abs(dy) < 2: return False
        return dy < 0 if lane == "RIGHT" else dy > 0

    def run(self):
        print(f"Processing video on {self.device}... Please wait.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            height, _ = frame.shape[:2]
            
            # Explicitly passing the device to the track method for better performance
            results = self.model.track(frame, persist=True, device=self.device, verbose=False)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for box, track_id, cls in zip(boxes, ids, classes):
                    cls = int(cls)
                    if cls not in self.class_map: continue

                    label = self.class_map[cls]
                    name = self.get_name(track_id, label)

                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    self.track_history.setdefault(track_id, []).append((cx, cy))
                    if len(self.track_history[track_id]) > 20:
                        self.track_history[track_id].pop(0)

                    lane = self.get_lane(cx, cy, height)
                    wrong = self.is_wrong_direction(track_id, lane)
                    color = (0, 0, 255) if wrong else (0, 255, 0)

                    # Draw graphics
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"WRONG {name}" if wrong else name
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.polylines(frame, [self.divider], True, (255, 0, 0), 2)
            
            self.out.write(frame)

            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        self.out.release() 
        cv2.destroyAllWindows()
        print("Video export complete.")

if __name__ == "__main__":
    video_path = r"video_path.mp4"
    detector = WrongDirectionDetector(video_path, "violation_output.mp4")
    detector.run()